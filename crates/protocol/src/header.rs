use std::sync::atomic::{AtomicU32, Ordering};

use crate::{Slot, HEADER_BYTES, NAME_BYTES, REASON_BYTES, SHM_MAGIC, SHM_VERSION};

/// One pass's overrides of the Model tab's values: `overrides` ([`crate::enums::pass_override`] bits)
/// names the fields the pass takes from here; the rest follow the Model tab.
#[repr(C)]
#[derive(Default)]
pub struct PassControl {
    pub overrides: AtomicU32,
    pub intensity_bits: AtomicU32,
    pub local_tone_bits: AtomicU32,
    pub local_structure_bits: AtomicU32,
    pub skin_structure_bits: AtomicU32,
    pub style: AtomicU32,
    pub auto_mask: AtomicU32,
}

impl PassControl {
    /// No overrides, and the Model tab's defaults in every field, so an override switched on later
    /// starts from what the Model tab does by default.
    pub fn reset_to_defaults(&self) {
        self.overrides.store(0, Ordering::Relaxed);
        self.intensity_bits.store(1.0f32.to_bits(), Ordering::Relaxed);
        self.local_tone_bits.store(1.0f32.to_bits(), Ordering::Relaxed);
        self.local_structure_bits.store(1.0f32.to_bits(), Ordering::Relaxed);
        self.skin_structure_bits.store((-1.0f32).to_bits(), Ordering::Relaxed);
        self.style.store(0, Ordering::Relaxed);
        self.auto_mask.store(1, Ordering::Relaxed);
    }

    /// `global` with this pass's overrides applied.
    pub fn apply(&self, global: Tuning) -> Tuning {
        use crate::enums::pass_override::*;
        let mask = self.overrides.load(Ordering::Relaxed);
        let f = |bit: u32, a: &AtomicU32, g: f32| if mask & bit != 0 { f32::from_bits(a.load(Ordering::Relaxed)) } else { g };
        let u = |bit: u32, a: &AtomicU32, g: u32| if mask & bit != 0 { a.load(Ordering::Relaxed) } else { g };
        Tuning {
            intensity: f(INTENSITY, &self.intensity_bits, global.intensity),
            local_tone: f(LOCAL_TONE, &self.local_tone_bits, global.local_tone),
            local_structure: f(LOCAL_STRUCTURE, &self.local_structure_bits, global.local_structure),
            skin_structure: f(SKIN_STRUCTURE, &self.skin_structure_bits, global.skin_structure),
            style: u(STYLE, &self.style, global.style),
            auto_mask: u(AUTO_MASK, &self.auto_mask, global.auto_mask),
        }
    }
}

impl PassControl {
    /// `global` with this pass's colour overrides (from `own`) applied.
    pub fn apply_colour(&self, own: &PassColour, global: EditColour) -> EditColour {
        use crate::enums::pass_override::*;
        let mask = self.overrides.load(Ordering::Relaxed);
        let f = |bit: u32, a: &AtomicU32, g: f32| if mask & bit != 0 { f32::from_bits(a.load(Ordering::Relaxed)) } else { g };
        EditColour {
            saturation: f(EDIT_SATURATION, &own.saturation_bits, global.saturation),
            contrast: f(EDIT_CONTRAST, &own.contrast_bits, global.contrast),
            detail: f(EDIT_DETAIL, &own.detail_bits, global.detail),
        }
    }
}

/// The network's passes as one frame runs them: how many, and each one's settings (the Model tab's
/// with that pass's overrides). Entries past `count` are the Model tab's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PassTunings {
    pub count: usize,
    pub tunings: [Tuning; crate::MAX_PASSES],
    /// How much of a pass's answer becomes the next pass's input (0 to 1; 1 is the answer itself).
    pub feedback: f32,
    /// Each pass's model resolution before the upscaler, a scale of DLSS's input size
    /// (`preupscale_scale`, or the pass's own `pass<N>_scale` where it overrides it); 1 is full size.
    pub scales: [f32; crate::MAX_PASSES],
    /// The colour of each pass's edit and the stack's colour controls.
    pub colour: ColourSettings,
    /// Each pass's placement ([`ShmHeader::placements`]).
    pub placements: [u32; crate::MAX_PASSES],
}

impl PassTunings {
    /// One pass with `tuning`, at full size.
    pub fn single(tuning: Tuning) -> Self {
        Self { count: 1, tunings: [tuning; crate::MAX_PASSES], feedback: 1.0, scales: [1.0; crate::MAX_PASSES], colour: ColourSettings::DEFAULT, placements: [crate::enums::placement::AUTO; crate::MAX_PASSES] }
    }

    pub fn passes(&self) -> &[Tuning] {
        &self.tunings[..self.count]
    }
}

/// One pass's own values for the colour of the edit ([`ShmHeader::pass_colour`]); which of them it uses
/// is in its [`PassControl::overrides`] (`EDIT_SATURATION`, `EDIT_CONTRAST`, `EDIT_DETAIL`).
#[repr(C)]
#[derive(Default)]
pub struct PassColour {
    pub saturation_bits: AtomicU32,
    pub contrast_bits: AtomicU32,
    pub detail_bits: AtomicU32,
}

impl PassColour {
    /// The Model tab's defaults (neutral), so an override switched on later starts from them.
    pub fn reset_to_defaults(&self) {
        for a in [&self.saturation_bits, &self.contrast_bits, &self.detail_bits] {
            a.store(1.0f32.to_bits(), Ordering::Relaxed);
        }
    }
}

/// How one pass shapes the network's change (the edit: its answer minus its input): the scale of the
/// edit's chroma (`saturation`), of its luma (`contrast`) and of its high frequencies against its low ones
/// (`detail`). 1, 1, 1 leaves the edit as the network made it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EditColour {
    pub saturation: f32,
    pub contrast: f32,
    pub detail: f32,
}

impl EditColour {
    pub const NEUTRAL: Self = Self { saturation: 1.0, contrast: 1.0, detail: 1.0 };

    pub fn is_neutral(&self) -> bool {
        *self == Self::NEUTRAL
    }
}

/// The colour of the edit for every pass ([`ShmHeader::pass_tunings`]): each pass's own scales (the
/// Model tab's with that pass's overrides), and the controls for the whole stack: the gamut guard, the
/// dither of the 8-bit answer after the upscaler, and the look-up table on the last pass's result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColourSettings {
    pub each: [EditColour; crate::MAX_PASSES],
    /// Pull a change that would leave the source's range (encoded 0-1 per channel, so also the BT.709
    /// primaries) back along itself instead of clipping channels.
    pub gamut_guard: bool,
    /// An ordered dither of +-0.5 of the 8-bit step before the after-the-upscaler answer is quantised.
    pub dither: bool,
    /// The look-up table's strength (0: off) and the edit size at which it applies in full.
    pub lut_strength: f32,
    pub lut_adapt: f32,
    /// Bumped when `edit_lut` in `config.ini` names another file (or the same one changed).
    pub lut_seq: u32,
}

impl ColourSettings {
    /// Every control at its default: the edit as the network made it.
    pub const DEFAULT: Self = Self { each: [EditColour::NEUTRAL; crate::MAX_PASSES], gamut_guard: true, dither: false, lut_strength: 0.0, lut_adapt: 0.05, lut_seq: 0 };
}

/// The control mask's rules ([`ShmHeader::mask_settings`]), as plain values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaskSettings {
    pub sky: bool,
    pub sky_margin: f32,
    pub dark: bool,
    pub dark_threshold: f32,
    pub dark_intensity: f32,
    pub dark_structure: f32,
    pub motion: bool,
    pub motion_threshold: f32,
    pub motion_intensity: f32,
    pub motion_structure: f32,
}

impl MaskSettings {
    /// Every rule off: the mask is neutral and nothing reads it.
    pub const OFF: Self = Self {
        sky: false,
        sky_margin: 0.00001,
        dark: false,
        dark_threshold: 0.05,
        dark_intensity: 0.5,
        dark_structure: 0.5,
        motion: false,
        motion_threshold: 16.0,
        motion_intensity: 0.5,
        motion_structure: 0.5,
    };

    pub fn any(&self) -> bool {
        self.sky || self.dark || self.motion
    }
}

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
    /// 6 is the pre-upscaler edit cache's rejection mask, written into DLSS's colour input on cached
    /// holds (`preupscale_cache_apply.comp`); the post path draws 6 as 0.
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
    /// A pending capture request, `(id << 16) | frames` (`crate::capture::encode_request`), 0 for
    /// none: one frame is the one-shot pair, more a numbered series of the next presented frames
    /// (the layer's `series` module), at most `crate::capture::MAX_FRAMES`. Written only through
    /// `crate::capture::request`; the layer zeroes it when it takes the request and says what
    /// became of it in the capture result (v18).
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
    /// After the upscaler only: before it the edit cache sets its own cadence (docs/ARCHITECTURE.md
    /// 4.10).
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

    // --- v16 -----------------------------------------------------------------------------
    /// The wall-clock time (Unix seconds) the layer of the game that holds this channel saw
    /// `VK_ERROR_DEVICE_LOST`, or 0 when it has not. Written once, when the layer's latch fires
    /// (`note_vk`), and reset when a game's layer opens the channel, so it describes the latest game
    /// session only. A field of its own rather than `layer_reason`: that string is the native
    /// backend's status line and is rewritten whenever the status changes, and `doctor` and the
    /// Status tab need the loss to survive whatever is written after it.
    pub device_lost_at: AtomicU32,

    // --- v17 -----------------------------------------------------------------------------
    /// How many consecutive model-mode holds the layer should dump before the upscaler (colour
    /// before and after the model, motion vectors, depth, jitter, exposure; docs/ARCHITECTURE.md
    /// 4.9), or 0 for none. Set by `shmctl dump-cache`; the layer zeroes it
    /// when it takes the request (the dump then runs to its end or to the first missed hold).
    pub preupscale_dump_request: AtomicU32,
    /// The pre-upscaler edit cache (docs/ARCHITECTURE.md 4.10: the network on every hold while the
    /// camera is still, on every second one while it moves; every hold with
    /// `NEURAL_FORGE_EDIT_CACHE=off`): holds that ran the network (refresh holds) and holds that
    /// reused its last correction instead (cached holds), counted since the layer opened the channel.
    pub cache_refresh_holds: AtomicU32,
    pub cache_cached_holds: AtomicU32,
    /// The last cached hold's share of pixels whose carried correction was rejected (the frame no
    /// longer looked like the refresh hold there, or the motion left the frame), in units of 1/10000.
    pub cache_rejected: AtomicU32,
    /// f32 bits: the cache passes' GPU time per hold in milliseconds, the median of the layer's last
    /// summary window (the latest reading before the first).
    pub cache_gpu_ms_bits: AtomicU32,

    // --- v18 -----------------------------------------------------------------------------
    /// The next capture request id a requester takes (`crate::capture::request`); ids run 1 to
    /// [`crate::capture::MAX_ID`] and wrap.
    pub capture_next_id: AtomicU32,
    /// The id of a capture the requester wants stopped (`crate::capture::cancel`): a running series
    /// ends at its next present, its pairs so far kept. 0 for none.
    pub capture_cancel: AtomicU32,
    /// What became of the latest capture request the layer took ([`crate::capture::Outcome`]),
    /// guarded by its own sequence number like the free-text fields. Written by the layer only.
    pub capture_result_seq: AtomicU32,
    capture_result: [AtomicU32; crate::capture::RESULT_WORDS],

    // --- v19 -----------------------------------------------------------------------------
    /// How many times the network runs over one frame (1 to [`crate::MAX_PASSES`]): each pass after the
    /// first takes the previous pass's picture as its colour input. Default 1.
    pub passes: AtomicU32,
    /// Each pass's overrides of the Model tab's values ([`PassControl`]), pass 1 first.
    pub pass: [PassControl; crate::MAX_PASSES],

    // --- v20 -----------------------------------------------------------------------------
    /// Which extracted network the layer runs ([`crate::enums::network`]): the full one (0, the
    /// default) or the small one. The layer switches a running game over at its next frame
    /// (`preupscale::native::Loader::ready`).
    pub network: AtomicU32,
    /// The control mask before the upscaler ([`MaskSettings`]): per pixel, how much of the model's edit is
    /// kept (intensity) and how much structure it is given. `mask_sky` (default on) keeps the game's sky:
    /// no edit where the depth is within `mask_sky_margin` of the far plane, fading in by twice the margin.
    /// `mask_dark` (off) and `mask_motion` (off) fade intensity and structure toward their own values below
    /// a brightness (`mask_dark_threshold`, on the encoded 0-1 scale) and above a motion (pixels per frame).
    pub mask_sky: AtomicU32,
    pub mask_sky_margin_bits: AtomicU32,
    pub mask_dark: AtomicU32,
    pub mask_dark_threshold_bits: AtomicU32,
    pub mask_dark_intensity_bits: AtomicU32,
    pub mask_dark_structure_bits: AtomicU32,
    pub mask_motion: AtomicU32,
    pub mask_motion_threshold_bits: AtomicU32,
    pub mask_motion_intensity_bits: AtomicU32,
    pub mask_motion_structure_bits: AtomicU32,
    /// f32 bits: how much of a pass's answer becomes the next pass's input, 0 to 1 (the rest is the pass's
    /// own input). 1 (the default) is the answer itself.
    pub pass_feedback_bits: AtomicU32,
    /// Times the layer reset the network's history before the upscaler because the picture changed as a
    /// whole (a scene cut: the mean luma difference of two frames' thumbnails at or above the
    /// threshold), counted since the layer opened the channel.
    pub scene_cuts: AtomicU32,

    // --- v21 -----------------------------------------------------------------------------
    /// f32 bits: the model resolution before the upscaler, a scale of DLSS's input size
    /// ([`crate::enums::model_scale`]: Full 1, Quality 0.667, Balanced 0.58, Performance 0.5, Ultra
    /// Performance 0.333). Below 1 the network runs on a downscaled picture and its edit is brought back to
    /// full size (`preupscale::native`, docs/NATIVE_BACKEND.md, "Model resolution before the upscaler").
    /// Default 1: the network at DLSS's input size, as before.
    pub preupscale_scale_bits: AtomicU32,
    /// f32 bits: each pass's own model resolution (`pass<N>_scale`), used where the pass's `overrides` has
    /// [`crate::enums::pass_override::SCALE`]. Default 1.
    pub pass_scale_bits: [AtomicU32; crate::MAX_PASSES],
    /// The colour of the network's edit (docs/NATIVE_BACKEND.md, "The colour of the edit"), for every
    /// pass: f32 bits, the scale of the edit's chroma (`edit_saturation`), of its luma around zero
    /// (`edit_contrast`) and of its high frequencies against its low ones (`edit_detail`), 0 to 4, 1 (the
    /// default) as the network made it.
    pub edit_saturation_bits: AtomicU32,
    pub edit_contrast_bits: AtomicU32,
    pub edit_detail_bits: AtomicU32,
    /// 1 (the default): where the shaped edit would take a channel out of the encoded 0-1 range, the whole
    /// change is shortened along itself until it fits instead of the channel clipping.
    pub edit_gamut_guard: AtomicU32,
    /// 1: an ordered dither of +-0.5 of the 8-bit step before the after-the-upscaler answer is quantised.
    /// Default 0.
    pub edit_dither: AtomicU32,
    /// f32 bits: the look-up table (`edit_lut` in `config.ini`, a `.cube` file) on the last pass's result,
    /// 0 (the default, off) to 1, scaled per pixel by the edit's size over `edit_lut_adapt` (0.001 to 1,
    /// default 0.05), capped at 1: a pixel the network left alone keeps its colour.
    pub edit_lut_strength_bits: AtomicU32,
    pub edit_lut_adapt_bits: AtomicU32,
    /// Bumped by whoever changes `edit_lut` in `config.ini`: the layer reads the file again. Not a setting.
    pub edit_lut_seq: AtomicU32,
    /// Each pass's own colour values ([`PassColour`]), used where its `overrides` say so.
    pub pass_colour: [PassColour; crate::MAX_PASSES],
    /// Where the network runs in the frame ([`crate::enums::placement`]; docs/NATIVE_BACKEND.md, "Where the
    /// model runs"): `auto` (0, the default) before the upscaler, after DLSS while Ray Reconstruction runs and
    /// at present where neither hook exists; `before`, `upscaled` (after DLSS's output, before the HUD) and
    /// `present` (on the presented frame) where the game allows it.
    pub placement: AtomicU32,
    /// Each pass's own placement (`pass<N>_placement`), used where its `overrides` has
    /// [`crate::enums::pass_override::PLACEMENT`]. Default `auto`.
    pub pass_placement: [AtomicU32; crate::MAX_PASSES],
    /// 1 (the default): at present, find the HUD (pixels that stay put and sharp while the scene moves) and
    /// keep the edit off it ([`HudSettings`]).
    pub hud_detect: AtomicU32,
    /// How many of [`Self::hud_region`] are in use (0 to [`MAX_HUD_REGIONS`]). With them, `hud_regions` in
    /// `config.ini` (`x,y,w,h[,keep|soften[,strength]]` separated by `;`); not a numeric setting.
    pub hud_region_count: AtomicU32,
    /// The user's HUD regions at present, in the output's 0-1 coordinates ([`HudRegion`]).
    pub hud_region: [HudRegionWords; MAX_HUD_REGIONS],
}

/// The most HUD regions (`hud_regions`).
pub const MAX_HUD_REGIONS: usize = 8;

/// One HUD region as the header holds it ([`HudRegion`]): f32 bits of its left, top, width and height (0-1 of
/// the output), its mode ([`crate::enums::hud_mode`]) and f32 bits of its soften strength.
#[repr(C)]
#[derive(Default)]
pub struct HudRegionWords {
    pub x_bits: AtomicU32,
    pub y_bits: AtomicU32,
    pub w_bits: AtomicU32,
    pub h_bits: AtomicU32,
    pub mode: AtomicU32,
    pub strength_bits: AtomicU32,
}

/// A HUD region at present: a rectangle in the output's 0-1 coordinates where the network's edit is kept off
/// (`keep`: none inside, fading in over a few pixels outside) or scaled by `strength` (`soften`, 0-1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HudRegion {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub mode: u32,
    pub strength: f32,
}

impl HudRegion {
    /// A soften region's strength when its text names none.
    pub const DEFAULT_STRENGTH: f32 = 0.5;

    /// Within the frame: left and top in 0-1, width and height to its edges, the strength in 0-1. `None` for a
    /// value that is not a number, an unknown mode or an empty rectangle.
    pub fn clamped(self) -> Option<Self> {
        if ![self.x, self.y, self.w, self.h, self.strength].iter().all(|v| v.is_finite()) || self.mode > crate::enums::hud_mode::SOFTEN {
            return None;
        }
        let x = self.x.clamp(0.0, 1.0);
        let y = self.y.clamp(0.0, 1.0);
        let w = self.w.clamp(0.0, 1.0 - x);
        let h = self.h.clamp(0.0, 1.0 - y);
        (w > 0.0 && h > 0.0).then_some(Self { x, y, w, h, mode: self.mode, strength: self.strength.clamp(0.0, 1.0) })
    }

    /// `x,y,w,h` (keep), `x,y,w,h,keep`, `x,y,w,h,soften` or `x,y,w,h,soften,strength`.
    pub fn parse(text: &str) -> Option<Self> {
        let f: Vec<&str> = text.split(',').map(str::trim).collect();
        let n = |i: usize| f.get(i).and_then(|v| v.parse::<f32>().ok());
        let (x, y, w, h) = (n(0)?, n(1)?, n(2)?, n(3)?);
        let (mode, strength) = match (f.get(4).copied(), f.len()) {
            (None, 4) | (Some("keep"), 5) => (crate::enums::hud_mode::KEEP, 0.0),
            (Some("soften"), 5) => (crate::enums::hud_mode::SOFTEN, Self::DEFAULT_STRENGTH),
            (Some("soften"), 6) => (crate::enums::hud_mode::SOFTEN, n(5)?),
            _ => return None,
        };
        Self { x, y, w, h, mode, strength }.clamped()
    }

    /// Regions separated by `;` (empty text: none). `Err` names the first one that does not parse, or says
    /// there are more than [`MAX_HUD_REGIONS`].
    pub fn parse_list(text: &str) -> Result<Vec<Self>, String> {
        let regions: Vec<Self> = text
            .split(';')
            .map(str::trim)
            .filter(|r| !r.is_empty())
            .map(|r| Self::parse(r).ok_or_else(|| format!("{r:?} is not x,y,w,h[,keep|soften[,strength]] inside the frame")))
            .collect::<Result<_, _>>()?;
        if regions.len() > MAX_HUD_REGIONS {
            return Err(format!("at most {MAX_HUD_REGIONS} regions, got {}", regions.len()));
        }
        Ok(regions)
    }

    /// The text [`Self::parse`] reads back.
    pub fn text(&self) -> String {
        match self.mode {
            crate::enums::hud_mode::SOFTEN => format!("{},{},{},{},soften,{}", self.x, self.y, self.w, self.h, self.strength),
            _ => format!("{},{},{},{},keep", self.x, self.y, self.w, self.h),
        }
    }

    /// [`Self::text`] of each, separated by `;`.
    pub fn list_text(regions: &[Self]) -> String {
        regions.iter().map(Self::text).collect::<Vec<_>>().join(";")
    }
}

/// The HUD settings at present ([`ShmHeader::hud_settings`]).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct HudSettings {
    pub detect: bool,
    pub regions: Vec<HudRegion>,
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
const _: () = assert!(std::mem::size_of::<ShmHeader>() == 2524, "the header layout changed -- bump SHM_VERSION");
const _: () = assert!(std::mem::offset_of!(ShmHeader, placement) == 2280, "layout changed -- bump SHM_VERSION");
const _: () = assert!(std::mem::offset_of!(ShmHeader, pass_placement) == 2284, "layout changed -- bump SHM_VERSION");
const _: () = assert!(std::mem::offset_of!(ShmHeader, hud_detect) == 2324, "layout changed -- bump SHM_VERSION");
const _: () = assert!(std::mem::offset_of!(ShmHeader, hud_region_count) == 2328, "layout changed -- bump SHM_VERSION");
const _: () = assert!(std::mem::offset_of!(ShmHeader, hud_region) == 2332, "layout changed -- bump SHM_VERSION");
const _: () = assert!(std::mem::offset_of!(ShmHeader, edit_saturation_bits) == 2128, "layout changed -- bump SHM_VERSION");
const _: () = assert!(std::mem::offset_of!(ShmHeader, edit_lut_seq) == 2156, "layout changed -- bump SHM_VERSION");
const _: () = assert!(std::mem::offset_of!(ShmHeader, pass_colour) == 2160, "layout changed -- bump SHM_VERSION");
const _: () = assert!(std::mem::offset_of!(ShmHeader, preupscale_scale_bits) == 2084, "layout changed -- bump SHM_VERSION");
const _: () = assert!(std::mem::offset_of!(ShmHeader, pass_scale_bits) == 2088, "layout changed -- bump SHM_VERSION");
const _: () = assert!(std::mem::offset_of!(ShmHeader, pass_feedback_bits) == 2076, "layout changed -- bump SHM_VERSION");
const _: () = assert!(std::mem::offset_of!(ShmHeader, scene_cuts) == 2080, "layout changed -- bump SHM_VERSION");
const _: () = assert!(std::mem::offset_of!(ShmHeader, network) == 2032, "layout changed -- bump SHM_VERSION");
const _: () = assert!(std::mem::offset_of!(ShmHeader, mask_sky) == 2036, "layout changed -- bump SHM_VERSION");
const _: () = assert!(std::mem::offset_of!(ShmHeader, passes) == 1748, "layout changed -- bump SHM_VERSION");
const _: () = assert!(std::mem::offset_of!(ShmHeader, pass) == 1752, "layout changed -- bump SHM_VERSION");
const _: () = assert!(std::mem::offset_of!(ShmHeader, capture_next_id) == 688, "layout changed -- bump SHM_VERSION");
const _: () = assert!(std::mem::offset_of!(ShmHeader, capture_cancel) == 692, "layout changed -- bump SHM_VERSION");
const _: () = assert!(std::mem::offset_of!(ShmHeader, capture_result_seq) == 696, "layout changed -- bump SHM_VERSION");
const _: () = assert!(std::mem::offset_of!(ShmHeader, capture_result) == 700, "layout changed -- bump SHM_VERSION");
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
const _: () = assert!(std::mem::offset_of!(ShmHeader, device_lost_at) == 664, "layout changed -- bump SHM_VERSION");
const _: () = assert!(std::mem::offset_of!(ShmHeader, preupscale_dump_request) == 668, "layout changed -- bump SHM_VERSION");
const _: () = assert!(std::mem::offset_of!(ShmHeader, cache_refresh_holds) == 672, "layout changed -- bump SHM_VERSION");
const _: () = assert!(std::mem::offset_of!(ShmHeader, cache_gpu_ms_bits) == 684, "layout changed -- bump SHM_VERSION");
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
pub const SETTING_BOUNDS: [(&str, f32, f32); 176] = [
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
    ("debug_view", 0.0, 6.0),
    ("debug_scale", 0.1, 10.0),
    // How many times the network runs over one frame (`resolved_passes`).
    ("passes", 1.0, 10.0),
    // Each pass's overrides (`enums::pass_override` bits) and its own values for them.
    ("pass1_overrides", 0.0, 2047.0),
    ("pass1_intensity", 0.0, 4.0),
    ("pass1_local_tone", 0.0, 4.0),
    ("pass1_local_structure", 0.0, 4.0),
    ("pass1_skin_structure", -1.0, 4.0),
    ("pass1_style", 0.0, 2.0),
    ("pass1_auto_mask", 0.0, 1.0),
    ("pass2_overrides", 0.0, 2047.0),
    ("pass2_intensity", 0.0, 4.0),
    ("pass2_local_tone", 0.0, 4.0),
    ("pass2_local_structure", 0.0, 4.0),
    ("pass2_skin_structure", -1.0, 4.0),
    ("pass2_style", 0.0, 2.0),
    ("pass2_auto_mask", 0.0, 1.0),
    ("pass3_overrides", 0.0, 2047.0),
    ("pass3_intensity", 0.0, 4.0),
    ("pass3_local_tone", 0.0, 4.0),
    ("pass3_local_structure", 0.0, 4.0),
    ("pass3_skin_structure", -1.0, 4.0),
    ("pass3_style", 0.0, 2.0),
    ("pass3_auto_mask", 0.0, 1.0),
    ("pass4_overrides", 0.0, 2047.0),
    ("pass4_intensity", 0.0, 4.0),
    ("pass4_local_tone", 0.0, 4.0),
    ("pass4_local_structure", 0.0, 4.0),
    ("pass4_skin_structure", -1.0, 4.0),
    ("pass4_style", 0.0, 2.0),
    ("pass4_auto_mask", 0.0, 1.0),
    ("pass5_overrides", 0.0, 2047.0),
    ("pass5_intensity", 0.0, 4.0),
    ("pass5_local_tone", 0.0, 4.0),
    ("pass5_local_structure", 0.0, 4.0),
    ("pass5_skin_structure", -1.0, 4.0),
    ("pass5_style", 0.0, 2.0),
    ("pass5_auto_mask", 0.0, 1.0),
    ("pass6_overrides", 0.0, 2047.0),
    ("pass6_intensity", 0.0, 4.0),
    ("pass6_local_tone", 0.0, 4.0),
    ("pass6_local_structure", 0.0, 4.0),
    ("pass6_skin_structure", -1.0, 4.0),
    ("pass6_style", 0.0, 2.0),
    ("pass6_auto_mask", 0.0, 1.0),
    ("pass7_overrides", 0.0, 2047.0),
    ("pass7_intensity", 0.0, 4.0),
    ("pass7_local_tone", 0.0, 4.0),
    ("pass7_local_structure", 0.0, 4.0),
    ("pass7_skin_structure", -1.0, 4.0),
    ("pass7_style", 0.0, 2.0),
    ("pass7_auto_mask", 0.0, 1.0),
    ("pass8_overrides", 0.0, 2047.0),
    ("pass8_intensity", 0.0, 4.0),
    ("pass8_local_tone", 0.0, 4.0),
    ("pass8_local_structure", 0.0, 4.0),
    ("pass8_skin_structure", -1.0, 4.0),
    ("pass8_style", 0.0, 2.0),
    ("pass8_auto_mask", 0.0, 1.0),
    ("pass9_overrides", 0.0, 2047.0),
    ("pass9_intensity", 0.0, 4.0),
    ("pass9_local_tone", 0.0, 4.0),
    ("pass9_local_structure", 0.0, 4.0),
    ("pass9_skin_structure", -1.0, 4.0),
    ("pass9_style", 0.0, 2.0),
    ("pass9_auto_mask", 0.0, 1.0),
    ("pass10_overrides", 0.0, 2047.0),
    ("pass10_intensity", 0.0, 4.0),
    ("pass10_local_tone", 0.0, 4.0),
    ("pass10_local_structure", 0.0, 4.0),
    ("pass10_skin_structure", -1.0, 4.0),
    ("pass10_style", 0.0, 2.0),
    ("pass10_auto_mask", 0.0, 1.0),
    // `enums::network`: 0 the full network, 1 the small one.
    ("network", 0.0, (crate::enums::network::COUNT - 1) as f32),
    // The control mask (`MaskSettings`).
    ("mask_sky", 0.0, 1.0),
    ("mask_sky_margin", 1e-07, 0.01),
    ("mask_dark", 0.0, 1.0),
    ("mask_dark_threshold", 0.0, 1.0),
    ("mask_dark_intensity", 0.0, 1.0),
    ("mask_dark_structure", 0.0, 1.0),
    ("mask_motion", 0.0, 1.0),
    ("mask_motion_threshold", 0.5, 256.0),
    ("mask_motion_intensity", 0.0, 1.0),
    ("mask_motion_structure", 0.0, 1.0),
    // How much of a pass's answer becomes the next pass's input.
    ("pass_feedback", 0.0, 1.0),
    // The model resolution before the upscaler (`enums::model_scale`), and each pass's own.
    ("preupscale_scale", crate::enums::model_scale::MIN, 1.0),
    ("pass1_scale", crate::enums::model_scale::MIN, 1.0),
    ("pass2_scale", crate::enums::model_scale::MIN, 1.0),
    ("pass3_scale", crate::enums::model_scale::MIN, 1.0),
    ("pass4_scale", crate::enums::model_scale::MIN, 1.0),
    ("pass5_scale", crate::enums::model_scale::MIN, 1.0),
    ("pass6_scale", crate::enums::model_scale::MIN, 1.0),
    ("pass7_scale", crate::enums::model_scale::MIN, 1.0),
    ("pass8_scale", crate::enums::model_scale::MIN, 1.0),
    ("pass9_scale", crate::enums::model_scale::MIN, 1.0),
    ("pass10_scale", crate::enums::model_scale::MIN, 1.0),
    // The colour of the edit (`ColourSettings`).
    ("edit_saturation", 0.0, 4.0),
    ("edit_contrast", 0.0, 4.0),
    ("edit_detail", 0.0, 4.0),
    ("edit_gamut_guard", 0.0, 1.0),
    ("edit_dither", 0.0, 1.0),
    ("edit_lut_strength", 0.0, 1.0),
    ("edit_lut_adapt", 0.001, 1.0),
    // Each pass's own colour values (its `overrides` say which it uses).
    ("pass1_edit_saturation", 0.0, 4.0),
    ("pass1_edit_contrast", 0.0, 4.0),
    ("pass1_edit_detail", 0.0, 4.0),
    ("pass2_edit_saturation", 0.0, 4.0),
    ("pass2_edit_contrast", 0.0, 4.0),
    ("pass2_edit_detail", 0.0, 4.0),
    ("pass3_edit_saturation", 0.0, 4.0),
    ("pass3_edit_contrast", 0.0, 4.0),
    ("pass3_edit_detail", 0.0, 4.0),
    ("pass4_edit_saturation", 0.0, 4.0),
    ("pass4_edit_contrast", 0.0, 4.0),
    ("pass4_edit_detail", 0.0, 4.0),
    ("pass5_edit_saturation", 0.0, 4.0),
    ("pass5_edit_contrast", 0.0, 4.0),
    ("pass5_edit_detail", 0.0, 4.0),
    ("pass6_edit_saturation", 0.0, 4.0),
    ("pass6_edit_contrast", 0.0, 4.0),
    ("pass6_edit_detail", 0.0, 4.0),
    ("pass7_edit_saturation", 0.0, 4.0),
    ("pass7_edit_contrast", 0.0, 4.0),
    ("pass7_edit_detail", 0.0, 4.0),
    ("pass8_edit_saturation", 0.0, 4.0),
    ("pass8_edit_contrast", 0.0, 4.0),
    ("pass8_edit_detail", 0.0, 4.0),
    ("pass9_edit_saturation", 0.0, 4.0),
    ("pass9_edit_contrast", 0.0, 4.0),
    ("pass9_edit_detail", 0.0, 4.0),
    ("pass10_edit_saturation", 0.0, 4.0),
    ("pass10_edit_contrast", 0.0, 4.0),
    ("pass10_edit_detail", 0.0, 4.0),
    // Where the network runs (`enums::placement`), and each pass's own.
    ("placement", 0.0, (crate::enums::placement::COUNT - 1) as f32),
    ("pass1_placement", 0.0, (crate::enums::placement::COUNT - 1) as f32),
    ("pass2_placement", 0.0, (crate::enums::placement::COUNT - 1) as f32),
    ("pass3_placement", 0.0, (crate::enums::placement::COUNT - 1) as f32),
    ("pass4_placement", 0.0, (crate::enums::placement::COUNT - 1) as f32),
    ("pass5_placement", 0.0, (crate::enums::placement::COUNT - 1) as f32),
    ("pass6_placement", 0.0, (crate::enums::placement::COUNT - 1) as f32),
    ("pass7_placement", 0.0, (crate::enums::placement::COUNT - 1) as f32),
    ("pass8_placement", 0.0, (crate::enums::placement::COUNT - 1) as f32),
    ("pass9_placement", 0.0, (crate::enums::placement::COUNT - 1) as f32),
    ("pass10_placement", 0.0, (crate::enums::placement::COUNT - 1) as f32),
    // The HUD at present: its detection.
    ("hud_detect", 0.0, 1.0),
];

/// The persisted name of pass `pass`'s (0 is pass 1) `field` (`overrides`, `intensity`, `local_tone`,
/// `local_structure`, `skin_structure`, `style`, `auto_mask`, `scale`): `pass<N>_<field>`.
pub fn pass_setting(pass: usize, field: &str) -> Option<&'static str> {
    let name = format!("pass{}_{field}", pass + 1);
    SETTING_BOUNDS.iter().map(|&(n, ..)| n).find(|n| *n == name)
}

/// A model resolution setting's bits as the scale the layer uses: within
/// [`crate::enums::model_scale::MIN`] and 1, and 1 for anything that is not a number.
pub fn resolved_scale(bits: u32) -> f32 {
    let s = f32::from_bits(bits);
    if s.is_finite() { s.clamp(crate::enums::model_scale::MIN, 1.0) } else { 1.0 }
}

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
        // The magic goes last (and is withdrawn first): a process opening the channel while this
        // runs sees no magic, not a valid-looking header still full of another layout's values.
        // Openers also serialise on the file's lock (`mapping::with_init_lock`).
        self.magic.store(0, Ordering::Release);
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
        self.passes.store(1, Ordering::Relaxed);
        for p in &self.pass {
            p.reset_to_defaults();
        }
        self.network.store(crate::enums::network::FULL, Ordering::Relaxed);
        self.mask_sky.store(1, Ordering::Relaxed);
        self.mask_sky_margin_bits.store(0.00001f32.to_bits(), Ordering::Relaxed);
        self.mask_dark.store(0, Ordering::Relaxed);
        self.mask_dark_threshold_bits.store(0.05f32.to_bits(), Ordering::Relaxed);
        self.mask_dark_intensity_bits.store(0.5f32.to_bits(), Ordering::Relaxed);
        self.mask_dark_structure_bits.store(0.5f32.to_bits(), Ordering::Relaxed);
        self.mask_motion.store(0, Ordering::Relaxed);
        self.mask_motion_threshold_bits.store(16.0f32.to_bits(), Ordering::Relaxed);
        self.mask_motion_intensity_bits.store(0.5f32.to_bits(), Ordering::Relaxed);
        self.mask_motion_structure_bits.store(0.5f32.to_bits(), Ordering::Relaxed);
        self.pass_feedback_bits.store(1.0f32.to_bits(), Ordering::Relaxed);
        self.preupscale_scale_bits.store(1.0f32.to_bits(), Ordering::Relaxed);
        for s in &self.pass_scale_bits {
            s.store(1.0f32.to_bits(), Ordering::Relaxed);
        }
        self.edit_saturation_bits.store(1.0f32.to_bits(), Ordering::Relaxed);
        self.edit_contrast_bits.store(1.0f32.to_bits(), Ordering::Relaxed);
        self.edit_detail_bits.store(1.0f32.to_bits(), Ordering::Relaxed);
        self.edit_gamut_guard.store(1, Ordering::Relaxed);
        self.edit_dither.store(0, Ordering::Relaxed);
        self.edit_lut_strength_bits.store(0.0f32.to_bits(), Ordering::Relaxed);
        self.edit_lut_adapt_bits.store(0.05f32.to_bits(), Ordering::Relaxed);
        self.edit_lut_seq.store(0, Ordering::Relaxed);
        for p in &self.pass_colour {
            p.reset_to_defaults();
        }
        self.placement.store(crate::enums::placement::AUTO, Ordering::Relaxed);
        for p in &self.pass_placement {
            p.store(crate::enums::placement::AUTO, Ordering::Relaxed);
        }
        self.hud_detect.store(1, Ordering::Relaxed);
        self.store_hud_regions(&[]);

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
        self.device_lost_at.store(0, Ordering::Relaxed);
        for f in [&self.cache_refresh_holds, &self.cache_cached_holds, &self.cache_rejected, &self.cache_gpu_ms_bits] {
            f.store(0, Ordering::Relaxed);
        }
        self.scene_cuts.store(0, Ordering::Relaxed);
        self.capture_next_id.store(0, Ordering::Relaxed);
        self.capture_cancel.store(0, Ordering::Relaxed);
        self.set_capture_result_words(&[0; crate::capture::RESULT_WORDS]);
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
        self.preupscale_dump_request.store(0, Ordering::Relaxed);

        self.version.store(SHM_VERSION, Ordering::Relaxed);
        self.magic.store(SHM_MAGIC, Ordering::Release);
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
        let settings: Vec<(&str, u32)> = defaults.persisted_settings().into_iter().map(|(name, _, bits)| (name, bits)).collect();
        self.write_settings_and_hud(&settings, Some(&[]));
    }

    /// Writes several settings as one unit (`persisted_settings` names, raw bits, each clamped as
    /// [`Self::apply_persisted_setting`] does) and bumps `control_seq` once. `tuning_seq` is their
    /// sequence lock: odd while the values change, even again after, so [`Self::global_tuning`]
    /// never reads half of a preset or a profile. It only grows (by two per write), and stays
    /// non-zero once anything has written: that is how "not yet configured" (0) is read. Returns
    /// whether every name was a setting.
    pub fn write_settings(&self, settings: &[(&str, u32)]) -> bool {
        self.write_settings_and_hud(settings, None)
    }

    /// The HUD regions at present (`hud_regions`), written as one unit like [`Self::write_settings`]
    /// (each clamped by [`HudRegion::clamped`], one that is not dropped, at most [`MAX_HUD_REGIONS`]).
    pub fn set_hud_regions(&self, regions: &[HudRegion]) {
        self.write_settings_and_hud(&[], Some(regions));
    }

    /// [`Self::write_settings`] and, with `regions`, the HUD regions, in one unit.
    pub fn write_settings_and_hud(&self, settings: &[(&str, u32)], regions: Option<&[HudRegion]>) -> bool {
        // Entering is a compare-and-swap from an even value to the next odd one, so two writers
        // (the GUI and the CLI) take turns. One left odd by a writer that died mid-write is taken
        // over once it has stayed on that value for STUCK_FOR: the next odd value is ours.
        let mut stuck: Option<(u32, std::time::Instant)> = None;
        let odd = loop {
            let seq = self.tuning_seq.load(Ordering::Relaxed);
            let next = if seq & 1 == 0 {
                seq.wrapping_add(1)
            } else if stuck.is_some_and(|(v, at)| v == seq && at.elapsed() >= STUCK_FOR) {
                seq.wrapping_add(2)
            } else {
                if stuck.is_none_or(|(v, _)| v != seq) {
                    stuck = Some((seq, std::time::Instant::now()));
                }
                std::thread::yield_now();
                continue;
            };
            if self.tuning_seq.compare_exchange_weak(seq, next, Ordering::Acquire, Ordering::Relaxed).is_ok() {
                break next;
            }
        };
        std::sync::atomic::fence(Ordering::Release);
        let mut all = true;
        for (name, bits) in settings {
            all &= self.apply_persisted_setting(name, *bits);
        }
        if let Some(regions) = regions {
            self.store_hud_regions(regions);
        }
        // Only if this writer still owns the section: one taken over as dead while it was merely slow
        // must not move the sequence back under the writer that took over (that one finishes it).
        self.finish_settings_write(odd);
        self.control_seq.fetch_add(1, Ordering::Relaxed);
        all
    }

    /// Ends the write section entered at `odd`, only if this writer still owns it: one taken over as
    /// dead while it was merely slow must not move the sequence back under the writer that took over.
    fn finish_settings_write(&self, odd: u32) {
        let _ = self.tuning_seq.compare_exchange(odd, odd.wrapping_add(1).max(2), Ordering::Release, Ordering::Relaxed);
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
    pub fn persisted_settings(&self) -> [(&'static str, bool, u32); 176] {
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
            ("passes", false, self.passes.load(Ordering::Relaxed)),
            ("pass1_overrides", false, self.pass[0].overrides.load(Ordering::Relaxed)),
            ("pass1_intensity", true, self.pass[0].intensity_bits.load(Ordering::Relaxed)),
            ("pass1_local_tone", true, self.pass[0].local_tone_bits.load(Ordering::Relaxed)),
            ("pass1_local_structure", true, self.pass[0].local_structure_bits.load(Ordering::Relaxed)),
            ("pass1_skin_structure", true, self.pass[0].skin_structure_bits.load(Ordering::Relaxed)),
            ("pass1_style", false, self.pass[0].style.load(Ordering::Relaxed)),
            ("pass1_auto_mask", false, self.pass[0].auto_mask.load(Ordering::Relaxed)),
            ("pass2_overrides", false, self.pass[1].overrides.load(Ordering::Relaxed)),
            ("pass2_intensity", true, self.pass[1].intensity_bits.load(Ordering::Relaxed)),
            ("pass2_local_tone", true, self.pass[1].local_tone_bits.load(Ordering::Relaxed)),
            ("pass2_local_structure", true, self.pass[1].local_structure_bits.load(Ordering::Relaxed)),
            ("pass2_skin_structure", true, self.pass[1].skin_structure_bits.load(Ordering::Relaxed)),
            ("pass2_style", false, self.pass[1].style.load(Ordering::Relaxed)),
            ("pass2_auto_mask", false, self.pass[1].auto_mask.load(Ordering::Relaxed)),
            ("pass3_overrides", false, self.pass[2].overrides.load(Ordering::Relaxed)),
            ("pass3_intensity", true, self.pass[2].intensity_bits.load(Ordering::Relaxed)),
            ("pass3_local_tone", true, self.pass[2].local_tone_bits.load(Ordering::Relaxed)),
            ("pass3_local_structure", true, self.pass[2].local_structure_bits.load(Ordering::Relaxed)),
            ("pass3_skin_structure", true, self.pass[2].skin_structure_bits.load(Ordering::Relaxed)),
            ("pass3_style", false, self.pass[2].style.load(Ordering::Relaxed)),
            ("pass3_auto_mask", false, self.pass[2].auto_mask.load(Ordering::Relaxed)),
            ("pass4_overrides", false, self.pass[3].overrides.load(Ordering::Relaxed)),
            ("pass4_intensity", true, self.pass[3].intensity_bits.load(Ordering::Relaxed)),
            ("pass4_local_tone", true, self.pass[3].local_tone_bits.load(Ordering::Relaxed)),
            ("pass4_local_structure", true, self.pass[3].local_structure_bits.load(Ordering::Relaxed)),
            ("pass4_skin_structure", true, self.pass[3].skin_structure_bits.load(Ordering::Relaxed)),
            ("pass4_style", false, self.pass[3].style.load(Ordering::Relaxed)),
            ("pass4_auto_mask", false, self.pass[3].auto_mask.load(Ordering::Relaxed)),
            ("pass5_overrides", false, self.pass[4].overrides.load(Ordering::Relaxed)),
            ("pass5_intensity", true, self.pass[4].intensity_bits.load(Ordering::Relaxed)),
            ("pass5_local_tone", true, self.pass[4].local_tone_bits.load(Ordering::Relaxed)),
            ("pass5_local_structure", true, self.pass[4].local_structure_bits.load(Ordering::Relaxed)),
            ("pass5_skin_structure", true, self.pass[4].skin_structure_bits.load(Ordering::Relaxed)),
            ("pass5_style", false, self.pass[4].style.load(Ordering::Relaxed)),
            ("pass5_auto_mask", false, self.pass[4].auto_mask.load(Ordering::Relaxed)),
            ("pass6_overrides", false, self.pass[5].overrides.load(Ordering::Relaxed)),
            ("pass6_intensity", true, self.pass[5].intensity_bits.load(Ordering::Relaxed)),
            ("pass6_local_tone", true, self.pass[5].local_tone_bits.load(Ordering::Relaxed)),
            ("pass6_local_structure", true, self.pass[5].local_structure_bits.load(Ordering::Relaxed)),
            ("pass6_skin_structure", true, self.pass[5].skin_structure_bits.load(Ordering::Relaxed)),
            ("pass6_style", false, self.pass[5].style.load(Ordering::Relaxed)),
            ("pass6_auto_mask", false, self.pass[5].auto_mask.load(Ordering::Relaxed)),
            ("pass7_overrides", false, self.pass[6].overrides.load(Ordering::Relaxed)),
            ("pass7_intensity", true, self.pass[6].intensity_bits.load(Ordering::Relaxed)),
            ("pass7_local_tone", true, self.pass[6].local_tone_bits.load(Ordering::Relaxed)),
            ("pass7_local_structure", true, self.pass[6].local_structure_bits.load(Ordering::Relaxed)),
            ("pass7_skin_structure", true, self.pass[6].skin_structure_bits.load(Ordering::Relaxed)),
            ("pass7_style", false, self.pass[6].style.load(Ordering::Relaxed)),
            ("pass7_auto_mask", false, self.pass[6].auto_mask.load(Ordering::Relaxed)),
            ("pass8_overrides", false, self.pass[7].overrides.load(Ordering::Relaxed)),
            ("pass8_intensity", true, self.pass[7].intensity_bits.load(Ordering::Relaxed)),
            ("pass8_local_tone", true, self.pass[7].local_tone_bits.load(Ordering::Relaxed)),
            ("pass8_local_structure", true, self.pass[7].local_structure_bits.load(Ordering::Relaxed)),
            ("pass8_skin_structure", true, self.pass[7].skin_structure_bits.load(Ordering::Relaxed)),
            ("pass8_style", false, self.pass[7].style.load(Ordering::Relaxed)),
            ("pass8_auto_mask", false, self.pass[7].auto_mask.load(Ordering::Relaxed)),
            ("pass9_overrides", false, self.pass[8].overrides.load(Ordering::Relaxed)),
            ("pass9_intensity", true, self.pass[8].intensity_bits.load(Ordering::Relaxed)),
            ("pass9_local_tone", true, self.pass[8].local_tone_bits.load(Ordering::Relaxed)),
            ("pass9_local_structure", true, self.pass[8].local_structure_bits.load(Ordering::Relaxed)),
            ("pass9_skin_structure", true, self.pass[8].skin_structure_bits.load(Ordering::Relaxed)),
            ("pass9_style", false, self.pass[8].style.load(Ordering::Relaxed)),
            ("pass9_auto_mask", false, self.pass[8].auto_mask.load(Ordering::Relaxed)),
            ("pass10_overrides", false, self.pass[9].overrides.load(Ordering::Relaxed)),
            ("pass10_intensity", true, self.pass[9].intensity_bits.load(Ordering::Relaxed)),
            ("pass10_local_tone", true, self.pass[9].local_tone_bits.load(Ordering::Relaxed)),
            ("pass10_local_structure", true, self.pass[9].local_structure_bits.load(Ordering::Relaxed)),
            ("pass10_skin_structure", true, self.pass[9].skin_structure_bits.load(Ordering::Relaxed)),
            ("pass10_style", false, self.pass[9].style.load(Ordering::Relaxed)),
            ("pass10_auto_mask", false, self.pass[9].auto_mask.load(Ordering::Relaxed)),
            ("network", false, self.network.load(Ordering::Relaxed)),
            ("mask_sky", false, self.mask_sky.load(Ordering::Relaxed)),
            ("mask_sky_margin", true, self.mask_sky_margin_bits.load(Ordering::Relaxed)),
            ("mask_dark", false, self.mask_dark.load(Ordering::Relaxed)),
            ("mask_dark_threshold", true, self.mask_dark_threshold_bits.load(Ordering::Relaxed)),
            ("mask_dark_intensity", true, self.mask_dark_intensity_bits.load(Ordering::Relaxed)),
            ("mask_dark_structure", true, self.mask_dark_structure_bits.load(Ordering::Relaxed)),
            ("mask_motion", false, self.mask_motion.load(Ordering::Relaxed)),
            ("mask_motion_threshold", true, self.mask_motion_threshold_bits.load(Ordering::Relaxed)),
            ("mask_motion_intensity", true, self.mask_motion_intensity_bits.load(Ordering::Relaxed)),
            ("mask_motion_structure", true, self.mask_motion_structure_bits.load(Ordering::Relaxed)),
            ("pass_feedback", true, self.pass_feedback_bits.load(Ordering::Relaxed)),
            ("preupscale_scale", true, self.preupscale_scale_bits.load(Ordering::Relaxed)),
            ("pass1_scale", true, self.pass_scale_bits[0].load(Ordering::Relaxed)),
            ("pass2_scale", true, self.pass_scale_bits[1].load(Ordering::Relaxed)),
            ("pass3_scale", true, self.pass_scale_bits[2].load(Ordering::Relaxed)),
            ("pass4_scale", true, self.pass_scale_bits[3].load(Ordering::Relaxed)),
            ("pass5_scale", true, self.pass_scale_bits[4].load(Ordering::Relaxed)),
            ("pass6_scale", true, self.pass_scale_bits[5].load(Ordering::Relaxed)),
            ("pass7_scale", true, self.pass_scale_bits[6].load(Ordering::Relaxed)),
            ("pass8_scale", true, self.pass_scale_bits[7].load(Ordering::Relaxed)),
            ("pass9_scale", true, self.pass_scale_bits[8].load(Ordering::Relaxed)),
            ("pass10_scale", true, self.pass_scale_bits[9].load(Ordering::Relaxed)),
            ("edit_saturation", true, self.edit_saturation_bits.load(Ordering::Relaxed)),
            ("edit_contrast", true, self.edit_contrast_bits.load(Ordering::Relaxed)),
            ("edit_detail", true, self.edit_detail_bits.load(Ordering::Relaxed)),
            ("edit_gamut_guard", false, self.edit_gamut_guard.load(Ordering::Relaxed)),
            ("edit_dither", false, self.edit_dither.load(Ordering::Relaxed)),
            ("edit_lut_strength", true, self.edit_lut_strength_bits.load(Ordering::Relaxed)),
            ("edit_lut_adapt", true, self.edit_lut_adapt_bits.load(Ordering::Relaxed)),
            ("pass1_edit_saturation", true, self.pass_colour[0].saturation_bits.load(Ordering::Relaxed)),
            ("pass1_edit_contrast", true, self.pass_colour[0].contrast_bits.load(Ordering::Relaxed)),
            ("pass1_edit_detail", true, self.pass_colour[0].detail_bits.load(Ordering::Relaxed)),
            ("pass2_edit_saturation", true, self.pass_colour[1].saturation_bits.load(Ordering::Relaxed)),
            ("pass2_edit_contrast", true, self.pass_colour[1].contrast_bits.load(Ordering::Relaxed)),
            ("pass2_edit_detail", true, self.pass_colour[1].detail_bits.load(Ordering::Relaxed)),
            ("pass3_edit_saturation", true, self.pass_colour[2].saturation_bits.load(Ordering::Relaxed)),
            ("pass3_edit_contrast", true, self.pass_colour[2].contrast_bits.load(Ordering::Relaxed)),
            ("pass3_edit_detail", true, self.pass_colour[2].detail_bits.load(Ordering::Relaxed)),
            ("pass4_edit_saturation", true, self.pass_colour[3].saturation_bits.load(Ordering::Relaxed)),
            ("pass4_edit_contrast", true, self.pass_colour[3].contrast_bits.load(Ordering::Relaxed)),
            ("pass4_edit_detail", true, self.pass_colour[3].detail_bits.load(Ordering::Relaxed)),
            ("pass5_edit_saturation", true, self.pass_colour[4].saturation_bits.load(Ordering::Relaxed)),
            ("pass5_edit_contrast", true, self.pass_colour[4].contrast_bits.load(Ordering::Relaxed)),
            ("pass5_edit_detail", true, self.pass_colour[4].detail_bits.load(Ordering::Relaxed)),
            ("pass6_edit_saturation", true, self.pass_colour[5].saturation_bits.load(Ordering::Relaxed)),
            ("pass6_edit_contrast", true, self.pass_colour[5].contrast_bits.load(Ordering::Relaxed)),
            ("pass6_edit_detail", true, self.pass_colour[5].detail_bits.load(Ordering::Relaxed)),
            ("pass7_edit_saturation", true, self.pass_colour[6].saturation_bits.load(Ordering::Relaxed)),
            ("pass7_edit_contrast", true, self.pass_colour[6].contrast_bits.load(Ordering::Relaxed)),
            ("pass7_edit_detail", true, self.pass_colour[6].detail_bits.load(Ordering::Relaxed)),
            ("pass8_edit_saturation", true, self.pass_colour[7].saturation_bits.load(Ordering::Relaxed)),
            ("pass8_edit_contrast", true, self.pass_colour[7].contrast_bits.load(Ordering::Relaxed)),
            ("pass8_edit_detail", true, self.pass_colour[7].detail_bits.load(Ordering::Relaxed)),
            ("pass9_edit_saturation", true, self.pass_colour[8].saturation_bits.load(Ordering::Relaxed)),
            ("pass9_edit_contrast", true, self.pass_colour[8].contrast_bits.load(Ordering::Relaxed)),
            ("pass9_edit_detail", true, self.pass_colour[8].detail_bits.load(Ordering::Relaxed)),
            ("pass10_edit_saturation", true, self.pass_colour[9].saturation_bits.load(Ordering::Relaxed)),
            ("pass10_edit_contrast", true, self.pass_colour[9].contrast_bits.load(Ordering::Relaxed)),
            ("pass10_edit_detail", true, self.pass_colour[9].detail_bits.load(Ordering::Relaxed)),
            ("placement", false, self.placement.load(Ordering::Relaxed)),
            ("pass1_placement", false, self.pass_placement[0].load(Ordering::Relaxed)),
            ("pass2_placement", false, self.pass_placement[1].load(Ordering::Relaxed)),
            ("pass3_placement", false, self.pass_placement[2].load(Ordering::Relaxed)),
            ("pass4_placement", false, self.pass_placement[3].load(Ordering::Relaxed)),
            ("pass5_placement", false, self.pass_placement[4].load(Ordering::Relaxed)),
            ("pass6_placement", false, self.pass_placement[5].load(Ordering::Relaxed)),
            ("pass7_placement", false, self.pass_placement[6].load(Ordering::Relaxed)),
            ("pass8_placement", false, self.pass_placement[7].load(Ordering::Relaxed)),
            ("pass9_placement", false, self.pass_placement[8].load(Ordering::Relaxed)),
            ("pass10_placement", false, self.pass_placement[9].load(Ordering::Relaxed)),
            ("hud_detect", false, self.hud_detect.load(Ordering::Relaxed)),
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
            "passes" => &self.passes,
            "pass1_overrides" => &self.pass[0].overrides,
            "pass1_intensity" => &self.pass[0].intensity_bits,
            "pass1_local_tone" => &self.pass[0].local_tone_bits,
            "pass1_local_structure" => &self.pass[0].local_structure_bits,
            "pass1_skin_structure" => &self.pass[0].skin_structure_bits,
            "pass1_style" => &self.pass[0].style,
            "pass1_auto_mask" => &self.pass[0].auto_mask,
            "pass2_overrides" => &self.pass[1].overrides,
            "pass2_intensity" => &self.pass[1].intensity_bits,
            "pass2_local_tone" => &self.pass[1].local_tone_bits,
            "pass2_local_structure" => &self.pass[1].local_structure_bits,
            "pass2_skin_structure" => &self.pass[1].skin_structure_bits,
            "pass2_style" => &self.pass[1].style,
            "pass2_auto_mask" => &self.pass[1].auto_mask,
            "pass3_overrides" => &self.pass[2].overrides,
            "pass3_intensity" => &self.pass[2].intensity_bits,
            "pass3_local_tone" => &self.pass[2].local_tone_bits,
            "pass3_local_structure" => &self.pass[2].local_structure_bits,
            "pass3_skin_structure" => &self.pass[2].skin_structure_bits,
            "pass3_style" => &self.pass[2].style,
            "pass3_auto_mask" => &self.pass[2].auto_mask,
            "pass4_overrides" => &self.pass[3].overrides,
            "pass4_intensity" => &self.pass[3].intensity_bits,
            "pass4_local_tone" => &self.pass[3].local_tone_bits,
            "pass4_local_structure" => &self.pass[3].local_structure_bits,
            "pass4_skin_structure" => &self.pass[3].skin_structure_bits,
            "pass4_style" => &self.pass[3].style,
            "pass4_auto_mask" => &self.pass[3].auto_mask,
            "pass5_overrides" => &self.pass[4].overrides,
            "pass5_intensity" => &self.pass[4].intensity_bits,
            "pass5_local_tone" => &self.pass[4].local_tone_bits,
            "pass5_local_structure" => &self.pass[4].local_structure_bits,
            "pass5_skin_structure" => &self.pass[4].skin_structure_bits,
            "pass5_style" => &self.pass[4].style,
            "pass5_auto_mask" => &self.pass[4].auto_mask,
            "pass6_overrides" => &self.pass[5].overrides,
            "pass6_intensity" => &self.pass[5].intensity_bits,
            "pass6_local_tone" => &self.pass[5].local_tone_bits,
            "pass6_local_structure" => &self.pass[5].local_structure_bits,
            "pass6_skin_structure" => &self.pass[5].skin_structure_bits,
            "pass6_style" => &self.pass[5].style,
            "pass6_auto_mask" => &self.pass[5].auto_mask,
            "pass7_overrides" => &self.pass[6].overrides,
            "pass7_intensity" => &self.pass[6].intensity_bits,
            "pass7_local_tone" => &self.pass[6].local_tone_bits,
            "pass7_local_structure" => &self.pass[6].local_structure_bits,
            "pass7_skin_structure" => &self.pass[6].skin_structure_bits,
            "pass7_style" => &self.pass[6].style,
            "pass7_auto_mask" => &self.pass[6].auto_mask,
            "pass8_overrides" => &self.pass[7].overrides,
            "pass8_intensity" => &self.pass[7].intensity_bits,
            "pass8_local_tone" => &self.pass[7].local_tone_bits,
            "pass8_local_structure" => &self.pass[7].local_structure_bits,
            "pass8_skin_structure" => &self.pass[7].skin_structure_bits,
            "pass8_style" => &self.pass[7].style,
            "pass8_auto_mask" => &self.pass[7].auto_mask,
            "pass9_overrides" => &self.pass[8].overrides,
            "pass9_intensity" => &self.pass[8].intensity_bits,
            "pass9_local_tone" => &self.pass[8].local_tone_bits,
            "pass9_local_structure" => &self.pass[8].local_structure_bits,
            "pass9_skin_structure" => &self.pass[8].skin_structure_bits,
            "pass9_style" => &self.pass[8].style,
            "pass9_auto_mask" => &self.pass[8].auto_mask,
            "pass10_overrides" => &self.pass[9].overrides,
            "pass10_intensity" => &self.pass[9].intensity_bits,
            "pass10_local_tone" => &self.pass[9].local_tone_bits,
            "pass10_local_structure" => &self.pass[9].local_structure_bits,
            "pass10_skin_structure" => &self.pass[9].skin_structure_bits,
            "pass10_style" => &self.pass[9].style,
            "pass10_auto_mask" => &self.pass[9].auto_mask,
            "network" => &self.network,
            "mask_sky" => &self.mask_sky,
            "mask_sky_margin" => &self.mask_sky_margin_bits,
            "mask_dark" => &self.mask_dark,
            "mask_dark_threshold" => &self.mask_dark_threshold_bits,
            "mask_dark_intensity" => &self.mask_dark_intensity_bits,
            "mask_dark_structure" => &self.mask_dark_structure_bits,
            "mask_motion" => &self.mask_motion,
            "mask_motion_threshold" => &self.mask_motion_threshold_bits,
            "mask_motion_intensity" => &self.mask_motion_intensity_bits,
            "mask_motion_structure" => &self.mask_motion_structure_bits,
            "pass_feedback" => &self.pass_feedback_bits,
            "preupscale_scale" => &self.preupscale_scale_bits,
            "pass1_scale" => &self.pass_scale_bits[0],
            "pass2_scale" => &self.pass_scale_bits[1],
            "pass3_scale" => &self.pass_scale_bits[2],
            "pass4_scale" => &self.pass_scale_bits[3],
            "pass5_scale" => &self.pass_scale_bits[4],
            "pass6_scale" => &self.pass_scale_bits[5],
            "pass7_scale" => &self.pass_scale_bits[6],
            "pass8_scale" => &self.pass_scale_bits[7],
            "pass9_scale" => &self.pass_scale_bits[8],
            "pass10_scale" => &self.pass_scale_bits[9],
            "edit_saturation" => &self.edit_saturation_bits,
            "edit_contrast" => &self.edit_contrast_bits,
            "edit_detail" => &self.edit_detail_bits,
            "edit_gamut_guard" => &self.edit_gamut_guard,
            "edit_dither" => &self.edit_dither,
            "edit_lut_strength" => &self.edit_lut_strength_bits,
            "edit_lut_adapt" => &self.edit_lut_adapt_bits,
            "pass1_edit_saturation" => &self.pass_colour[0].saturation_bits,
            "pass1_edit_contrast" => &self.pass_colour[0].contrast_bits,
            "pass1_edit_detail" => &self.pass_colour[0].detail_bits,
            "pass2_edit_saturation" => &self.pass_colour[1].saturation_bits,
            "pass2_edit_contrast" => &self.pass_colour[1].contrast_bits,
            "pass2_edit_detail" => &self.pass_colour[1].detail_bits,
            "pass3_edit_saturation" => &self.pass_colour[2].saturation_bits,
            "pass3_edit_contrast" => &self.pass_colour[2].contrast_bits,
            "pass3_edit_detail" => &self.pass_colour[2].detail_bits,
            "pass4_edit_saturation" => &self.pass_colour[3].saturation_bits,
            "pass4_edit_contrast" => &self.pass_colour[3].contrast_bits,
            "pass4_edit_detail" => &self.pass_colour[3].detail_bits,
            "pass5_edit_saturation" => &self.pass_colour[4].saturation_bits,
            "pass5_edit_contrast" => &self.pass_colour[4].contrast_bits,
            "pass5_edit_detail" => &self.pass_colour[4].detail_bits,
            "pass6_edit_saturation" => &self.pass_colour[5].saturation_bits,
            "pass6_edit_contrast" => &self.pass_colour[5].contrast_bits,
            "pass6_edit_detail" => &self.pass_colour[5].detail_bits,
            "pass7_edit_saturation" => &self.pass_colour[6].saturation_bits,
            "pass7_edit_contrast" => &self.pass_colour[6].contrast_bits,
            "pass7_edit_detail" => &self.pass_colour[6].detail_bits,
            "pass8_edit_saturation" => &self.pass_colour[7].saturation_bits,
            "pass8_edit_contrast" => &self.pass_colour[7].contrast_bits,
            "pass8_edit_detail" => &self.pass_colour[7].detail_bits,
            "pass9_edit_saturation" => &self.pass_colour[8].saturation_bits,
            "pass9_edit_contrast" => &self.pass_colour[8].contrast_bits,
            "pass9_edit_detail" => &self.pass_colour[8].detail_bits,
            "pass10_edit_saturation" => &self.pass_colour[9].saturation_bits,
            "pass10_edit_contrast" => &self.pass_colour[9].contrast_bits,
            "pass10_edit_detail" => &self.pass_colour[9].detail_bits,
            "placement" => &self.placement,
            "pass1_placement" => &self.pass_placement[0],
            "pass2_placement" => &self.pass_placement[1],
            "pass3_placement" => &self.pass_placement[2],
            "pass4_placement" => &self.pass_placement[3],
            "pass5_placement" => &self.pass_placement[4],
            "pass6_placement" => &self.pass_placement[5],
            "pass7_placement" => &self.pass_placement[6],
            "pass8_placement" => &self.pass_placement[7],
            "pass9_placement" => &self.pass_placement[8],
            "pass10_placement" => &self.pass_placement[9],
            "hud_detect" => &self.hud_detect,
            _ => return false,
        };
        field.store(bits, Ordering::Relaxed);
        true
    }

    pub fn neural_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed) != 0
    }

    /// The Model tab's settings, as the network uses them.
    /// Read under `tuning_seq` ([`Self::write_settings`]): a preset or profile being written is seen
    /// whole, before or after. A writer that died mid-write (the sequence left on one odd value) is
    /// recognised after a few thousand looks and the values are read as they stand.
    pub fn global_tuning(&self) -> Tuning {
        self.read_settings(|| self.global_tuning_now())
    }

    /// The pass count ([`Self::resolved_passes`]) and each pass's settings, read as one unit under
    /// `tuning_seq` like [`Self::global_tuning`].
    pub fn pass_tunings(&self) -> PassTunings {
        self.read_settings(|| {
            let global = self.global_tuning_now();
            let mut tunings = [global; crate::MAX_PASSES];
            for (t, p) in tunings.iter_mut().zip(&self.pass) {
                *t = p.apply(global);
            }
            let feedback = f32::from_bits(self.pass_feedback_bits.load(Ordering::Relaxed));
            let feedback = if feedback.is_finite() { feedback.clamp(0.0, 1.0) } else { 1.0 };
            let global_scale = resolved_scale(self.preupscale_scale_bits.load(Ordering::Relaxed));
            let mut scales = [global_scale; crate::MAX_PASSES];
            for (k, s) in scales.iter_mut().enumerate() {
                if self.pass[k].overrides.load(Ordering::Relaxed) & crate::enums::pass_override::SCALE != 0 {
                    *s = resolved_scale(self.pass_scale_bits[k].load(Ordering::Relaxed));
                }
            }
            PassTunings { count: self.resolved_passes() as usize, tunings, feedback, scales, colour: self.colour_now(), placements: self.placements_now() }
        })
    }

    /// The colour of the edit ([`ColourSettings`]), each value clamped to its range (a non-finite one is
    /// the default). Read inside [`Self::pass_tunings`]'s unit.
    fn colour_now(&self) -> ColourSettings {
        let f = |a: &AtomicU32, name: &str, default: f32| {
            let (lo, hi) = setting_bounds(name).expect("a colour setting");
            let v = f32::from_bits(a.load(Ordering::Relaxed));
            if v.is_finite() { v.clamp(lo, hi) } else { default }
        };
        let global = EditColour {
            saturation: f(&self.edit_saturation_bits, "edit_saturation", 1.0),
            contrast: f(&self.edit_contrast_bits, "edit_contrast", 1.0),
            detail: f(&self.edit_detail_bits, "edit_detail", 1.0),
        };
        let mut each = [global; crate::MAX_PASSES];
        for ((e, p), own) in each.iter_mut().zip(&self.pass).zip(&self.pass_colour) {
            let c = p.apply_colour(own, global);
            let g = |v: f32| if v.is_finite() { v.clamp(0.0, 4.0) } else { 1.0 };
            *e = EditColour { saturation: g(c.saturation), contrast: g(c.contrast), detail: g(c.detail) };
        }
        ColourSettings {
            each,
            gamut_guard: self.edit_gamut_guard.load(Ordering::Relaxed) != 0,
            dither: self.edit_dither.load(Ordering::Relaxed) != 0,
            lut_strength: f(&self.edit_lut_strength_bits, "edit_lut_strength", 0.0),
            lut_adapt: f(&self.edit_lut_adapt_bits, "edit_lut_adapt", 0.05),
            lut_seq: self.edit_lut_seq.load(Ordering::Relaxed),
        }
    }

    /// Stores the HUD regions as they stand (inside a write section, or on a header nobody reads yet).
    fn store_hud_regions(&self, regions: &[HudRegion]) {
        let kept: Vec<HudRegion> = regions.iter().filter_map(|r| r.clamped()).take(MAX_HUD_REGIONS).collect();
        for (k, w) in self.hud_region.iter().enumerate() {
            let r = kept.get(k).copied().unwrap_or(HudRegion { x: 0.0, y: 0.0, w: 0.0, h: 0.0, mode: crate::enums::hud_mode::KEEP, strength: HudRegion::DEFAULT_STRENGTH });
            w.x_bits.store(r.x.to_bits(), Ordering::Relaxed);
            w.y_bits.store(r.y.to_bits(), Ordering::Relaxed);
            w.w_bits.store(r.w.to_bits(), Ordering::Relaxed);
            w.h_bits.store(r.h.to_bits(), Ordering::Relaxed);
            w.mode.store(r.mode, Ordering::Relaxed);
            w.strength_bits.store(r.strength.to_bits(), Ordering::Relaxed);
        }
        self.hud_region_count.store(kept.len() as u32, Ordering::Relaxed);
    }

    /// The HUD settings at present: the detection switch and the regions in use (each clamped; one that is not
    /// a valid rectangle is left out), read as one unit like [`Self::global_tuning`].
    pub fn hud_settings(&self) -> HudSettings {
        self.read_settings(|| {
            let n = (self.hud_region_count.load(Ordering::Relaxed) as usize).min(MAX_HUD_REGIONS);
            let f = |a: &AtomicU32| f32::from_bits(a.load(Ordering::Relaxed));
            let regions = self.hud_region[..n]
                .iter()
                .filter_map(|w| HudRegion { x: f(&w.x_bits), y: f(&w.y_bits), w: f(&w.w_bits), h: f(&w.h_bits), mode: w.mode.load(Ordering::Relaxed), strength: f(&w.strength_bits) }.clamped())
                .collect();
            HudSettings { detect: self.hud_detect.load(Ordering::Relaxed) != 0, regions }
        })
    }

    /// The pass count and each pass's placement ([`crate::enums::placement`]: `placement`, or the pass's own
    /// where its `overrides` has [`crate::enums::pass_override::PLACEMENT`]; a value out of range is `auto`),
    /// read as one unit like [`Self::pass_tunings`].
    pub fn placements(&self) -> (usize, [u32; crate::MAX_PASSES]) {
        self.read_settings(|| (self.resolved_passes() as usize, self.placements_now()))
    }

    fn placements_now(&self) -> [u32; crate::MAX_PASSES] {
        let valid = |v: u32| if v < crate::enums::placement::COUNT { v } else { crate::enums::placement::AUTO };
        let global = valid(self.placement.load(Ordering::Relaxed));
        std::array::from_fn(|k| {
            if self.pass[k].overrides.load(Ordering::Relaxed) & crate::enums::pass_override::PLACEMENT != 0 {
                valid(self.pass_placement[k].load(Ordering::Relaxed))
            } else {
                global
            }
        })
    }

    /// The Network setting ([`crate::enums::network`]); a value out of range is the full network.
    pub fn network_choice(&self) -> u32 {
        let v = self.network.load(Ordering::Relaxed);
        if v < crate::enums::network::COUNT { v } else { crate::enums::network::FULL }
    }

    /// The control mask's rules, each value clamped to its range.
    pub fn mask_settings(&self) -> MaskSettings {
        let f = |a: &AtomicU32, name: &str| {
            let (lo, hi) = setting_bounds(name).expect("a mask setting");
            let v = f32::from_bits(a.load(Ordering::Relaxed));
            if v.is_finite() { v.clamp(lo, hi) } else { lo }
        };
        let on = |a: &AtomicU32| a.load(Ordering::Relaxed) != 0;
        self.read_settings(|| MaskSettings {
            sky: on(&self.mask_sky),
            sky_margin: f(&self.mask_sky_margin_bits, "mask_sky_margin"),
            dark: on(&self.mask_dark),
            dark_threshold: f(&self.mask_dark_threshold_bits, "mask_dark_threshold"),
            dark_intensity: f(&self.mask_dark_intensity_bits, "mask_dark_intensity"),
            dark_structure: f(&self.mask_dark_structure_bits, "mask_dark_structure"),
            motion: on(&self.mask_motion),
            motion_threshold: f(&self.mask_motion_threshold_bits, "mask_motion_threshold"),
            motion_intensity: f(&self.mask_motion_intensity_bits, "mask_motion_intensity"),
            motion_structure: f(&self.mask_motion_structure_bits, "mask_motion_structure"),
        })
    }

    /// The Passes setting, clamped to 1 to [`crate::MAX_PASSES`].
    pub fn resolved_passes(&self) -> u32 {
        self.passes.load(Ordering::Relaxed).clamp(1, crate::MAX_PASSES as u32)
    }

    fn global_tuning_now(&self) -> Tuning {
        Tuning {
            intensity: f32::from_bits(self.intensity_bits.load(Ordering::Relaxed)),
            local_tone: f32::from_bits(self.local_tone_bits.load(Ordering::Relaxed)),
            local_structure: f32::from_bits(self.local_structure_bits.load(Ordering::Relaxed)),
            skin_structure: f32::from_bits(self.skin_structure_bits.load(Ordering::Relaxed)),
            style: self.style.load(Ordering::Relaxed),
            auto_mask: self.auto_mask.load(Ordering::Relaxed),
        }
    }

    /// `read` under `tuning_seq` ([`Self::write_settings`]): a preset or profile being written is seen
    /// whole, before or after.
    fn read_settings<T>(&self, read: impl Fn() -> T) -> T {
        // A live writer moves the sequence on within microseconds; one that stays on the same odd
        // value for STUCK_FOR has died mid-write. That value is remembered (per process), so later
        // reads do not wait for it again; the next write takes it over.
        let mut stuck: Option<(u32, std::time::Instant)> = None;
        loop {
            let before = self.tuning_seq.load(Ordering::Acquire);
            if before & 1 == 1 && DEAD_TUNING_SEQ.load(Ordering::Relaxed) == before {
                return read();
            }
            if before & 1 == 0 {
                let tuning = read();
                std::sync::atomic::fence(Ordering::Acquire);
                if self.tuning_seq.load(Ordering::Relaxed) == before {
                    return tuning;
                }
            } else if let Some((value, at)) = stuck.filter(|(v, _)| *v == before) {
                if at.elapsed() >= STUCK_FOR {
                    DEAD_TUNING_SEQ.store(value, Ordering::Relaxed);
                    return read();
                }
                std::thread::yield_now();
                continue;
            } else {
                stuck = Some((before, std::time::Instant::now()));
            }
            std::hint::spin_loop();
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
    /// The capture result's raw words, written as one unit (`crate::capture::publish`).
    pub(crate) fn set_capture_result_words(&self, words: &[u32; crate::capture::RESULT_WORDS]) {
        store_words_seq_guarded(&self.capture_result_seq, &self.capture_result, words);
    }
    /// The capture result's raw words as one consistent snapshot, `None` while a write keeps
    /// changing them (`crate::capture::result`).
    pub(crate) fn capture_result_words(&self) -> Option<[u32; crate::capture::RESULT_WORDS]> {
        load_words_seq_guarded(&self.capture_result_seq, &self.capture_result)
    }
}

/// How long one odd `tuning_seq` must stay unchanged before its writer counts as dead (a live
/// writer moves it on within microseconds, a preempted one within a scheduler slice or two).
const STUCK_FOR: std::time::Duration = std::time::Duration::from_millis(50);

/// The odd `tuning_seq` this process found left by a dead writer: reads skip the wait for it.
static DEAD_TUNING_SEQ: AtomicU32 = AtomicU32::new(0);

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
    // Cut at a character boundary: half a UTF-8 sequence would read back as U+FFFD.
    let mut n = bytes.len().min(W * 4 - 1);
    while !s.is_char_boundary(n) {
        n -= 1;
    }
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

/// [`store_seq_guarded`] for a block of plain words.
fn store_words_seq_guarded<const W: usize>(seq: &AtomicU32, words: &[AtomicU32; W], values: &[u32; W]) {
    let odd = seq.load(Ordering::Relaxed) | 1;
    seq.store(odd, Ordering::Relaxed);
    std::sync::atomic::fence(Ordering::Release);
    for (word, &value) in words.iter().zip(values) {
        word.store(value, Ordering::Relaxed);
    }
    seq.store(odd.wrapping_add(1), Ordering::Release);
}

/// [`load_seq_guarded`] for a block of plain words: `None` when no consistent snapshot was seen.
fn load_words_seq_guarded<const W: usize>(seq: &AtomicU32, words: &[AtomicU32; W]) -> Option<[u32; W]> {
    for _ in 0..64 {
        let before = seq.load(Ordering::Acquire);
        if before & 1 != 0 {
            std::hint::spin_loop();
            continue;
        }
        let snapshot: [u32; W] = std::array::from_fn(|i| words[i].load(Ordering::Relaxed));
        std::sync::atomic::fence(Ordering::Acquire);
        if seq.load(Ordering::Relaxed) == before {
            return Some(snapshot);
        }
    }
    None
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
        h.preupscale_dump_request.store(16, Ordering::Relaxed);
        h.init_defaults();
        assert_eq!(h.preupscale_dump_request.load(Ordering::Relaxed), 0, "a stale cache-dump request does not survive");
        assert_eq!(h.magic.load(Ordering::Relaxed), SHM_MAGIC);
        assert!(h.is_valid());
        assert!(h.neural_enabled());
        assert_eq!(f32::from_bits(h.intensity_bits.load(Ordering::Relaxed)), 1.0);
        // The composition is on by default; bypass is an explicit debug choice.
        assert_eq!(h.composition_bypass.load(Ordering::Relaxed), 0);
        // Slot 1 (v3) starts idle, same shape as slot 0.
        assert_eq!(h.seq_req_b.load(Ordering::Relaxed), 0);
        assert_eq!(h.seq_resp_b.load(Ordering::Relaxed), 0);
        assert_eq!(h.proxy_format_b.load(Ordering::Relaxed), crate::enums::proxy_format::RGBA8);
        assert_eq!(h.device_lost_at.load(Ordering::Relaxed), 0, "a fresh session has not lost its device");
    }

    #[test]
    fn settings_written_together_are_read_together() {
        let h = std::sync::Arc::new(ShmHeader::default());
        h.init_defaults();
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let writer = {
            let (h, stop) = (h.clone(), stop.clone());
            std::thread::spawn(move || {
                let mut i = 0u32;
                while !stop.load(Ordering::Relaxed) {
                    let v = if i.is_multiple_of(2) { 0.5f32 } else { 2.0 };
                    h.write_settings(&[("intensity", v.to_bits()), ("local_tone", v.to_bits()), ("local_structure", v.to_bits())]);
                    i += 1;
                }
            })
        };
        for _ in 0..20_000 {
            let t = h.global_tuning();
            assert!(t.intensity == t.local_tone && t.local_tone == t.local_structure, "a torn read: {t:?}");
        }
        stop.store(true, Ordering::Relaxed);
        writer.join().unwrap();
        assert_eq!(h.tuning_seq.load(Ordering::Relaxed) % 2, 0);
        assert!(!h.write_settings(&[("no_such_setting", 1)]));
        // A writer that died mid-write: the sequence stays odd, and the values are still read,
        // the second time without waiting; the next write takes the section over.
        let dead = h.tuning_seq.load(Ordering::Relaxed) | 1;
        h.tuning_seq.store(dead, Ordering::Relaxed);
        let _ = h.global_tuning();
        assert_eq!(DEAD_TUNING_SEQ.load(Ordering::Relaxed), dead);
        let _ = h.global_tuning();
        assert!(h.write_settings(&[("intensity", 0.75f32.to_bits())]));
        assert_eq!(h.tuning_seq.load(Ordering::Relaxed), dead + 3);
        assert_eq!(h.global_tuning().intensity, 0.75);
    }

    #[test]
    fn a_writer_taken_over_does_not_move_the_sequence_back() {
        let h = ShmHeader::default();
        h.init_defaults();
        // A slow writer holds 5; another found it dead and took over with 7, then finished at 8.
        h.tuning_seq.store(8, Ordering::Relaxed);
        // The slow writer finishing now must leave 8 alone (its own end would be 6).
        h.finish_settings_write(5);
        assert_eq!(h.tuning_seq.load(Ordering::Relaxed), 8);
        // And a write started after it still works from 8.
        assert!(h.write_settings(&[("intensity", 1.5f32.to_bits())]));
        assert_eq!(h.tuning_seq.load(Ordering::Relaxed), 10);
    }

    #[test]
    fn two_writers_take_turns() {
        let h = std::sync::Arc::new(ShmHeader::default());
        h.init_defaults();
        let start = h.tuning_seq.load(Ordering::Relaxed);
        std::thread::scope(|s| {
            for w in 0..2 {
                let h = &h;
                s.spawn(move || {
                    for _ in 0..5000 {
                        let v = if w == 0 { 0.5f32 } else { 2.0 };
                        h.write_settings(&[("intensity", v.to_bits()), ("local_tone", v.to_bits())]);
                    }
                });
            }
            for _ in 0..20_000 {
                let t = h.global_tuning();
                assert_eq!(t.intensity, t.local_tone, "a torn read");
            }
        });
        assert_eq!(h.tuning_seq.load(Ordering::Relaxed), start + 2 * 2 * 5000, "every write took its own turn");
    }

    #[test]
    fn long_text_is_cut_at_a_character_boundary() {
        let h = ShmHeader::default();
        let name = "é".repeat(200);
        h.set_game_name(&name);
        let back = h.game_name();
        assert!(!back.is_empty() && !back.contains('\u{FFFD}'), "{back:?}");
        assert!(back.chars().all(|c| c == 'é'));
    }

    #[test]
    fn device_lost_at_is_appended_after_native_running() {
        // v16 appends one word: every earlier field keeps its offset, and the new one sits last.
        assert_eq!(std::mem::offset_of!(ShmHeader, device_lost_at), std::mem::offset_of!(ShmHeader, native_running) + 4);
    }

    #[test]
    fn preupscale_dump_request_is_appended_after_device_lost_at() {
        // v17 appends words: every earlier field keeps its offset, the cache's status fields sit last.
        assert_eq!(std::mem::offset_of!(ShmHeader, preupscale_dump_request), std::mem::offset_of!(ShmHeader, device_lost_at) + 4);
        assert_eq!(std::mem::offset_of!(ShmHeader, cache_refresh_holds), std::mem::offset_of!(ShmHeader, preupscale_dump_request) + 4);
        assert_eq!(std::mem::offset_of!(ShmHeader, cache_cached_holds), std::mem::offset_of!(ShmHeader, cache_refresh_holds) + 4);
        assert_eq!(std::mem::offset_of!(ShmHeader, cache_rejected), std::mem::offset_of!(ShmHeader, cache_cached_holds) + 4);
        assert_eq!(std::mem::offset_of!(ShmHeader, cache_gpu_ms_bits), std::mem::offset_of!(ShmHeader, cache_rejected) + 4);
        assert_eq!(std::mem::offset_of!(ShmHeader, capture_next_id), std::mem::offset_of!(ShmHeader, cache_gpu_ms_bits) + 4);
        let h = ShmHeader::default();
        h.init_defaults();
        assert_eq!(h.preupscale_dump_request.load(Ordering::Relaxed), 0, "a fresh session dumps nothing");
    }

    #[test]
    fn the_control_mask_is_appended_after_the_passes() {
        // v20 appends the mask's rules: every earlier field keeps its offset.
        assert_eq!(std::mem::offset_of!(ShmHeader, mask_sky), std::mem::offset_of!(ShmHeader, network) + 4);
        assert_eq!(std::mem::offset_of!(ShmHeader, mask_motion_structure_bits) + 4, std::mem::offset_of!(ShmHeader, pass_feedback_bits));
        assert_eq!(std::mem::offset_of!(ShmHeader, pass_feedback_bits) + 4, std::mem::offset_of!(ShmHeader, scene_cuts));
        assert_eq!(crate::SHM_VERSION, 21);
        let h = ShmHeader::default();
        h.init_defaults();
        let m = h.mask_settings();
        assert!(m.sky && !m.dark && !m.motion, "keeping the sky is the only rule on by default");
        assert_eq!(m, MaskSettings { sky: true, ..MaskSettings::OFF });
        assert!(h.write_settings(&[("mask_motion", 1), ("mask_motion_threshold", 1000.0f32.to_bits())]));
        assert_eq!(h.mask_settings().motion_threshold, 256.0, "clamped to its range");
    }

    #[test]
    fn the_passes_are_appended_after_the_capture_result() {
        // v19 appends the pass count and the per-pass overrides: every earlier field keeps its offset.
        assert_eq!(std::mem::offset_of!(ShmHeader, passes), std::mem::offset_of!(ShmHeader, capture_result) + crate::capture::RESULT_WORDS * 4);
        assert_eq!(std::mem::offset_of!(ShmHeader, pass), std::mem::offset_of!(ShmHeader, passes) + 4);
        assert_eq!(std::mem::size_of::<PassControl>(), 28);
        assert_eq!(std::mem::offset_of!(ShmHeader, pass) + crate::MAX_PASSES * 28, std::mem::offset_of!(ShmHeader, network));
    }

    #[test]
    fn the_network_is_appended_after_the_passes() {
        // v20 appends the Network setting: every earlier field keeps its offset.
        assert_eq!(std::mem::offset_of!(ShmHeader, network), std::mem::offset_of!(ShmHeader, pass) + crate::MAX_PASSES * 28);
        assert_eq!(std::mem::offset_of!(ShmHeader, network) + 4, std::mem::offset_of!(ShmHeader, mask_sky));
        assert_eq!(crate::SHM_VERSION, 21);
        let h = ShmHeader::default();
        h.init_defaults();
        assert_eq!(h.network.load(Ordering::Relaxed), crate::enums::network::FULL, "the full network is the default");
    }

    #[test]
    fn the_scene_cut_counter_is_appended_after_the_passes() {
        // v20's last word: every earlier field keeps its offset, and v21's fields follow it.
        assert_eq!(std::mem::offset_of!(ShmHeader, scene_cuts), std::mem::offset_of!(ShmHeader, pass_feedback_bits) + 4);
        assert_eq!(std::mem::offset_of!(ShmHeader, scene_cuts) + 4, std::mem::offset_of!(ShmHeader, preupscale_scale_bits));
        assert_eq!(crate::SHM_VERSION, 21);
        let h = ShmHeader::default();
        h.scene_cuts.store(5, Ordering::Relaxed);
        h.init_defaults();
        assert_eq!(h.scene_cuts.load(Ordering::Relaxed), 0, "a fresh session has seen no cut");
    }

    #[test]
    fn the_colour_of_the_edit_is_appended_after_the_model_resolution() {
        // v21 appends the colour of the edit after the model resolution: every earlier field keeps its offset,
        // the per-pass values last.
        assert_eq!(std::mem::offset_of!(ShmHeader, edit_saturation_bits), std::mem::offset_of!(ShmHeader, pass_scale_bits) + crate::MAX_PASSES * 4);
        assert_eq!(std::mem::size_of::<PassColour>(), 12);
        assert_eq!(std::mem::offset_of!(ShmHeader, pass_colour), std::mem::offset_of!(ShmHeader, edit_lut_seq) + 4);
        assert_eq!(std::mem::offset_of!(ShmHeader, pass_colour) + crate::MAX_PASSES * 12, std::mem::offset_of!(ShmHeader, placement));
        assert_eq!(crate::SHM_VERSION, 21);
        let h = ShmHeader::default();
        h.edit_lut_seq.store(9, Ordering::Relaxed);
        h.init_defaults();
        let c = h.pass_tunings().colour;
        assert_eq!(c, ColourSettings::DEFAULT, "neutral scales, the guard on, no dither and no look-up table");
        assert!(c.each.iter().all(EditColour::is_neutral));
    }

    #[test]
    fn a_pass_takes_only_the_colour_values_it_overrides() {
        use crate::enums::pass_override::*;
        let h = ShmHeader::default();
        h.init_defaults();
        assert!(h.write_settings(&[
            ("edit_saturation", 0.5f32.to_bits()),
            ("edit_detail", 2.0f32.to_bits()),
            ("pass2_edit_saturation", 1.5f32.to_bits()),
            ("pass2_edit_contrast", 0.25f32.to_bits()),
            ("pass2_overrides", EDIT_SATURATION),
            ("edit_lut_adapt", 0.0f32.to_bits()),
        ]));
        let c = h.pass_tunings().colour;
        assert_eq!(c.each[0], EditColour { saturation: 0.5, contrast: 1.0, detail: 2.0 }, "pass 1 follows the Model tab");
        assert_eq!(c.each[1], EditColour { saturation: 1.5, contrast: 1.0, detail: 2.0 }, "pass 2's contrast is not overridden");
        assert_eq!(c.lut_adapt, 0.001, "clamped to its range");
        h.edit_contrast_bits.store(f32::NAN.to_bits(), Ordering::Relaxed);
        assert_eq!(h.pass_tunings().colour.each[0].contrast, 1.0, "a non-finite value is the default");
    }

    #[test]
    fn one_pass_following_the_model_tab_is_the_default() {
        let h = ShmHeader::default();
        h.passes.store(7, Ordering::Relaxed);
        h.pass[3].overrides.store(crate::enums::pass_override::ALL, Ordering::Relaxed);
        h.pass[3].intensity_bits.store(0.25f32.to_bits(), Ordering::Relaxed);
        h.init_defaults();
        let t = h.pass_tunings();
        assert_eq!(t.count, 1);
        assert_eq!(t.feedback, 1.0, "the next pass takes the answer itself by default");
        assert_eq!(t.passes(), &[h.global_tuning()]);
        assert!(t.tunings.iter().all(|p| *p == h.global_tuning()), "no pass overrides anything");
    }

    #[test]
    fn a_pass_takes_only_the_values_it_overrides() {
        use crate::enums::pass_override::*;
        let h = ShmHeader::default();
        h.init_defaults();
        h.intensity_bits.store(0.8f32.to_bits(), Ordering::Relaxed);
        h.local_tone_bits.store(1.5f32.to_bits(), Ordering::Relaxed);
        h.passes.store(3, Ordering::Relaxed);
        let p = &h.pass[1];
        p.intensity_bits.store(0.3f32.to_bits(), Ordering::Relaxed);
        p.local_tone_bits.store(0.1f32.to_bits(), Ordering::Relaxed);
        p.style.store(2, Ordering::Relaxed);
        p.overrides.store(INTENSITY | STYLE, Ordering::Relaxed);
        let t = h.pass_tunings();
        assert_eq!(t.count, 3);
        assert_eq!(t.tunings[0], h.global_tuning());
        assert_eq!(t.tunings[1], Tuning { intensity: 0.3, style: 2, ..h.global_tuning() }, "local tone is not overridden");
        assert_eq!(t.tunings[2], h.global_tuning());
    }

    #[test]
    fn the_pass_count_is_clamped_to_one_to_max_passes() {
        let h = ShmHeader::default();
        h.init_defaults();
        for (stored, resolved) in [(0, 1), (1, 1), (10, 10), (11, 10), (u32::MAX, 10)] {
            h.passes.store(stored, Ordering::Relaxed);
            assert_eq!(h.resolved_passes(), resolved, "{stored}");
        }
        assert_eq!(crate::setting_bounds("passes"), Some((1.0, crate::MAX_PASSES as f32)));
        assert!(h.apply_persisted_setting("passes", 40));
        assert_eq!(h.passes.load(Ordering::Relaxed), 10);
    }

    #[test]
    fn every_pass_setting_is_persisted_by_name() {
        let h = ShmHeader::default();
        h.init_defaults();
        assert!(h.write_settings(&[("pass10_overrides", 1), ("pass10_intensity", 0.5f32.to_bits()), ("passes", 10)]));
        assert_eq!(h.pass[9].overrides.load(Ordering::Relaxed), 1);
        assert_eq!(h.pass_tunings().tunings[9].intensity, 0.5);
        assert!(!h.apply_persisted_setting("pass11_intensity", 0), "there is no eleventh pass");
        assert!(h.apply_persisted_setting("pass1_overrides", 4095));
        assert_eq!(h.pass[0].overrides.load(Ordering::Relaxed), crate::enums::pass_override::ALL, "clamped to the eleven fields");
    }

    #[test]
    fn the_placement_and_the_hud_are_appended_after_the_colour_of_the_edit() {
        use crate::enums::{pass_override, placement};
        assert_eq!(std::mem::offset_of!(ShmHeader, placement), std::mem::offset_of!(ShmHeader, pass_colour) + crate::MAX_PASSES * 12);
        assert_eq!(std::mem::offset_of!(ShmHeader, pass_placement), std::mem::offset_of!(ShmHeader, placement) + 4);
        assert_eq!(std::mem::offset_of!(ShmHeader, hud_detect), std::mem::offset_of!(ShmHeader, pass_placement) + crate::MAX_PASSES * 4);
        assert_eq!(std::mem::offset_of!(ShmHeader, hud_region_count), std::mem::offset_of!(ShmHeader, hud_detect) + 4);
        assert_eq!(std::mem::offset_of!(ShmHeader, hud_region), std::mem::offset_of!(ShmHeader, hud_region_count) + 4);
        assert_eq!(std::mem::offset_of!(ShmHeader, hud_region) + MAX_HUD_REGIONS * 24, std::mem::size_of::<ShmHeader>());
        let h = ShmHeader::default();
        h.init_defaults();
        assert_eq!(h.placements(), (1, [placement::AUTO; crate::MAX_PASSES]), "auto for every pass by default");
        assert!(h.write_settings(&[("passes", 3), ("placement", placement::BEFORE), ("pass2_placement", placement::UPSCALED), ("pass3_placement", placement::PRESENT), ("pass2_overrides", pass_override::PLACEMENT)]));
        let (count, p) = h.placements();
        assert_eq!((count, &p[..3]), (3, &[placement::BEFORE, placement::UPSCALED, placement::BEFORE][..]), "pass 3 does not override its placement");
        assert_eq!(h.pass_tunings().placements, p);
        assert!(h.apply_persisted_setting("placement", 9));
        assert_eq!(h.placement.load(Ordering::Relaxed), placement::PRESENT, "clamped to the last placement");
        assert_eq!(placement::parse("upscaled"), Some(placement::UPSCALED));
        assert_eq!(placement::parse("2"), Some(placement::UPSCALED));
        assert_eq!(placement::parse("later"), None);
    }

    #[test]
    fn hud_regions_parse_clamp_and_print() {
        use crate::enums::hud_mode;
        let r = HudRegion::parse_list("0.1,0.2,0.3,0.4;0.5,0.5,0.9,0.9,soften; 0,0,1,0.1,soften,0.25 ;0.2,0.2,0.1,0.1,keep").unwrap();
        assert_eq!(r[0], HudRegion { x: 0.1, y: 0.2, w: 0.3, h: 0.4, mode: hud_mode::KEEP, strength: 0.0 });
        assert_eq!(r[1], HudRegion { x: 0.5, y: 0.5, w: 0.5, h: 0.5, mode: hud_mode::SOFTEN, strength: HudRegion::DEFAULT_STRENGTH }, "clamped to the frame");
        assert_eq!(r[2].strength, 0.25);
        assert_eq!(r[3].mode, hud_mode::KEEP);
        assert_eq!(HudRegion::parse_list("").unwrap(), vec![]);
        for bad in ["0.1,0.2,0.3", "0.1,0.2,0.3,0.4,blur", "0.1,0.2,0,0.4", "nan,0,1,1", "0.1,0.2,0.3,0.4,keep,0.5"] {
            assert!(HudRegion::parse_list(bad).is_err(), "{bad}");
        }
        assert!(HudRegion::parse_list(&["0,0,0.1,0.1"; MAX_HUD_REGIONS + 1].join(";")).is_err());
        assert_eq!(HudRegion::parse_list(&HudRegion::list_text(&r)).unwrap(), r, "the text reads back");
        let h = ShmHeader::default();
        h.init_defaults();
        h.set_hud_regions(&r);
        assert_eq!(h.hud_settings(), HudSettings { detect: true, regions: r });
        h.hud_region_count.store(99, Ordering::Relaxed);
        assert_eq!(h.hud_settings().regions.len(), 4, "at most the regions held: the empty slots are left out");
    }

    #[test]
    fn the_model_resolution_is_appended_last_and_full_size_by_default() {
        // v21 appends the model resolution and each pass's own: every earlier field keeps its offset.
        assert_eq!(std::mem::offset_of!(ShmHeader, preupscale_scale_bits), std::mem::offset_of!(ShmHeader, scene_cuts) + 4);
        assert_eq!(std::mem::offset_of!(ShmHeader, pass_scale_bits), std::mem::offset_of!(ShmHeader, preupscale_scale_bits) + 4);
        assert_eq!(std::mem::offset_of!(ShmHeader, pass_scale_bits) + crate::MAX_PASSES * 4, std::mem::offset_of!(ShmHeader, edit_saturation_bits));
        assert_eq!(crate::SHM_VERSION, 21);
        let h = ShmHeader::default();
        h.preupscale_scale_bits.store(0.5f32.to_bits(), Ordering::Relaxed);
        h.pass_scale_bits[2].store(0.5f32.to_bits(), Ordering::Relaxed);
        h.init_defaults();
        let t = h.pass_tunings();
        assert_eq!(t.scales, [1.0; crate::MAX_PASSES], "every pass at full size by default");
        assert_eq!(f32::from_bits(h.pass_scale_bits[2].load(Ordering::Relaxed)), 1.0);
    }

    #[test]
    fn a_pass_takes_its_own_model_resolution_only_when_it_overrides_it() {
        use crate::enums::pass_override::SCALE;
        let h = ShmHeader::default();
        h.init_defaults();
        assert!(h.write_settings(&[("preupscale_scale", 0.667f32.to_bits()), ("pass2_scale", 0.5f32.to_bits()), ("pass3_scale", 0.333f32.to_bits())]));
        assert!(h.write_settings(&[("pass2_overrides", SCALE), ("passes", 3)]));
        let t = h.pass_tunings();
        assert_eq!(t.scales[0], 0.667, "pass 1 follows the setting");
        assert_eq!(t.scales[1], 0.5, "pass 2 overrides it");
        assert_eq!(t.scales[2], 0.667, "pass 3 has a value of its own but does not override");
        assert!(h.write_settings(&[("preupscale_scale", 0.01f32.to_bits())]));
        assert_eq!(h.pass_tunings().scales[0], crate::enums::model_scale::MIN, "clamped to its range");
        h.preupscale_scale_bits.store(f32::NAN.to_bits(), Ordering::Relaxed);
        assert_eq!(h.pass_tunings().scales[0], 1.0, "not a number is full size");
        assert_eq!(crate::enums::model_scale::preset_of(0.58), Some(2));
        assert_eq!(crate::enums::model_scale::preset_of(0.7), None);
    }

    #[test]
    fn the_capture_result_is_appended_after_the_cache_status() {
        // v18 appends the capture request ids and the result block: every earlier field keeps its offset.
        assert_eq!(std::mem::offset_of!(ShmHeader, capture_next_id), std::mem::offset_of!(ShmHeader, cache_gpu_ms_bits) + 4);
        assert_eq!(std::mem::offset_of!(ShmHeader, capture_cancel), std::mem::offset_of!(ShmHeader, capture_next_id) + 4);
        assert_eq!(std::mem::offset_of!(ShmHeader, capture_result_seq), std::mem::offset_of!(ShmHeader, capture_cancel) + 4);
        assert_eq!(std::mem::offset_of!(ShmHeader, capture_result), std::mem::offset_of!(ShmHeader, capture_result_seq) + 4);
        assert_eq!(std::mem::offset_of!(ShmHeader, passes), std::mem::offset_of!(ShmHeader, capture_result) + crate::capture::RESULT_WORDS * 4);
        let h = ShmHeader::default();
        h.capture_cancel.store(3, Ordering::Relaxed);
        h.set_capture_result_words(&[7; crate::capture::RESULT_WORDS]);
        h.init_defaults();
        assert_eq!(h.capture_cancel.load(Ordering::Relaxed), 0);
        assert_eq!(h.capture_result_words(), Some([0; crate::capture::RESULT_WORDS]), "a fresh session has no capture result");
    }

    #[test]
    fn a_word_block_reader_never_returns_a_write_in_progress() {
        let h = ShmHeader::default();
        h.set_capture_result_words(&[5; crate::capture::RESULT_WORDS]);
        assert_eq!(h.capture_result_seq.load(Ordering::Relaxed) & 1, 0);
        let even = h.capture_result_seq.load(Ordering::Relaxed);
        h.capture_result_seq.store(even | 1, Ordering::Release);
        assert_eq!(h.capture_result_words(), None, "an odd sequence is a write in progress");
        h.capture_result_seq.store(even + 2, Ordering::Release);
        assert_eq!(h.capture_result_words(), Some([5; crate::capture::RESULT_WORDS]));
    }

    #[test]
    fn init_defaults_clears_a_device_loss_and_a_settings_reset_keeps_it() {
        let h = ShmHeader::default();
        h.init_defaults();
        h.device_lost_at.store(1_760_000_000, Ordering::Relaxed);
        // A settings reset is not a new session: the loss is still the latest game's.
        h.reset_persisted_settings();
        assert_eq!(h.device_lost_at.load(Ordering::Relaxed), 1_760_000_000);
        h.init_defaults();
        assert_eq!(h.device_lost_at.load(Ordering::Relaxed), 0);
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

    #[test]
    fn the_network_setting_defaults_to_the_full_network_and_clamps() {
        let h = ShmHeader::default();
        h.init_defaults();
        assert_eq!(h.network_choice(), crate::enums::network::FULL);
        assert!(h.write_settings(&[("network", crate::enums::network::SMALL)]));
        assert_eq!(h.network_choice(), crate::enums::network::SMALL);
        // A stored value past the range (clamped by every writer) still reads as a network.
        assert!(h.write_settings(&[("network", 7)]));
        assert_eq!(h.network.load(Ordering::Relaxed), crate::enums::network::SMALL, "the write clamps");
        h.network.store(7, Ordering::Relaxed);
        assert_eq!(h.network_choice(), crate::enums::network::FULL);
        h.reset_persisted_settings();
        assert_eq!(h.network_choice(), crate::enums::network::FULL, "Reset goes back to the full network");
        assert_eq!((crate::enums::network::name(0), crate::enums::network::name(1)), ("full", "small"));
    }
}
