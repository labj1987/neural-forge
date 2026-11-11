//! DLSS's own parameters, as NGX was given them: the records Neural Forge's NGX wrapper for Proton
//! (`crates/ngx-wrapper`, docs/PROTON_NGX.md) writes into `/dev/shm/neural-forge-ngx-<pid>.ring` in the same
//! process, one per creation and evaluate of a DLSS feature.
//!
//! The records are the only source of what DLSS works on (`preupscale.rs`): which images are its colour input, depth,
//! motion vectors, output and exposure image, the camera jitter, the motion vectors' units, the game's reset, the
//! pre-exposure, whether the colour input is HDR and whether the feature is Ray Reconstruction. A launch belongs to
//! an evaluate when it names the evaluate's colour input between the evaluate's record and its end record
//! ([`evaluate_naming`], [`ended`]). Without a ring (no wrapper, `NEURAL_FORGE_NGX_RING=off`) nothing is held before
//! or after DLSS, and the status says so ([`no_wrapper`]). The counts of records read and of evaluates matched to
//! launches are published on the channel (`ngx_*` in `shmctl status`).
//!
//! The ring's format and its writer's protocol are in `crates/ngx-wrapper/src/ring.rs`; [`tests`] check the
//! offsets read here against it.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use ash::vk;
use ash::vk::Handle;

/// The file's first 8 bytes, version, and sizes (`crates/ngx-wrapper/src/ring.rs`).
pub(crate) const MAGIC: [u8; 8] = *b"NFNGXRG1";
pub(crate) const VERSION: u32 = 2;
pub(crate) const HEADER_BYTES: usize = 32;
pub(crate) const RECORD_BYTES: usize = 128;

/// `api`'s kind bit of an end-of-evaluate record (`crates/ngx-wrapper/src/ring.rs`: `KIND_END`), and the API's bits.
pub(crate) const KIND_END: u32 = 1 << 16;
pub(crate) const API_MASK: u32 = 0xffff;

/// [`Record::api`] of a D3D11 evaluate (the wrapper's `ring::API_D3D11`).
pub(crate) const API_D3D11: u32 = 11;

/// NGX's feature numbers.
pub(crate) const FEATURE_SR: u32 = 1;
pub(crate) const FEATURE_RR: u32 = 13;

/// `DLSS.Feature.Create.Flags`' IsHDR bit.
pub(crate) const FLAG_HDR: u32 = 1;

/// The environment variable: `off` reads no ring; a path reads that file instead of this process's.
pub(crate) const RING_ENV: &str = "NEURAL_FORGE_NGX_RING";

/// One record ([`parse_record`]).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Record {
    pub seq: u64,
    /// Milliseconds since the wrapper loaded, when the record was written.
    pub t_ms: f64,
    /// The API (`api`'s low bits).
    pub api: u32,
    /// An end-of-evaluate record: written once NVIDIA's `EvaluateFeature` returned, with the evaluate's own fields.
    pub end: bool,
    /// NGX's handle id and the calling thread: with the feature and the images, what ties an end to its evaluate.
    pub handle: u32,
    pub thread: u32,
    pub feature: u32,
    pub create_flags: u32,
    pub colour: vk::Image,
    pub mvec: vk::Image,
    pub depth: vk::Image,
    pub output: vk::Image,
    pub exposure: vk::Image,
    /// Render pixels.
    pub jitter: [f32; 2],
    /// What turns the motion vectors into render pixels (`MV.Scale`).
    pub mv_scale: [f32; 2],
    /// 0 when the game gave none.
    pub pre_exposure: f32,
    pub exposure_scale: f32,
    pub reset: bool,
    pub colour_extent: (u32, u32),
    pub output_extent: (u32, u32),
}

impl Record {
    /// An evaluate (a creation names no colour input), not its end.
    pub(crate) fn is_evaluate(&self) -> bool {
        self.colour != vk::Image::null() && !self.end
    }
    /// Whether `end` is this evaluate's end record: written after it, for the same feature, handle, thread and images.
    pub(crate) fn ended_by(&self, end: &Record) -> bool {
        end.end
            && end.seq > self.seq
            && (end.feature, end.handle, end.thread) == (self.feature, self.handle, self.thread)
            && [end.colour, end.mvec, end.depth, end.output, end.exposure] == [self.colour, self.mvec, self.depth, self.output, self.exposure]
    }
    pub(crate) fn is_rr(&self) -> bool {
        self.feature == FEATURE_RR
    }
    /// Super Resolution or Ray Reconstruction: the features whose colour input the hold works on.
    pub(crate) fn upscales(&self) -> bool {
        matches!(self.feature, FEATURE_SR | FEATURE_RR)
    }
    pub(crate) fn hdr(&self) -> bool {
        self.create_flags & FLAG_HDR != 0
    }
    /// What turns the motion vectors into the colour input's pixels (`MV.Scale`), when usable: finite, magnitude in
    /// [1e-3, 1e5].
    pub(crate) fn mv_units(&self) -> Option<[f32; 2]> {
        self.mv_scale.iter().all(|s| s.is_finite() && (1e-3..=1e5).contains(&s.abs())).then_some(self.mv_scale)
    }
    /// What the HDR encode's paper white is multiplied by so the game's exposure applies to the scene as DLSS
    /// would: the colour input is the scene times the pre-exposure, and DLSS multiplies the exposure image's value
    /// by the exposure scale. 1 when the game gives neither.
    pub(crate) fn paper_white_factor(&self) -> f32 {
        let pre = if self.pre_exposure.is_finite() && self.pre_exposure > 0.0 { self.pre_exposure } else { 1.0 };
        let scale = if self.exposure_scale.is_finite() && self.exposure_scale > 0.0 { self.exposure_scale } else { 1.0 };
        let f = pre / scale;
        if f.is_finite() && (1e-4..=1e4).contains(&f) {
            f
        } else {
            1.0
        }
    }
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().expect("4 bytes"))
}
fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().expect("8 bytes"))
}
fn f32_at(b: &[u8], at: usize) -> f32 {
    f32::from_bits(u32_at(b, at))
}

/// The header's slot count, if `b` starts with a ring header this build reads.
pub(crate) fn parse_header(b: &[u8]) -> Result<u32, String> {
    if b.len() < HEADER_BYTES || b[..8] != MAGIC {
        return Err("not an NGX ring".into());
    }
    let (version, record, slots) = (u32_at(b, 8), u32_at(b, 12), u32_at(b, 16));
    if version != VERSION || record as usize != RECORD_BYTES {
        return Err(format!("ring version {version} with {record}-byte records; this layer reads version {VERSION}"));
    }
    if slots == 0 || b.len() < HEADER_BYTES + slots as usize * RECORD_BYTES {
        return Err(format!("ring of {slots} slots in {} bytes", b.len()));
    }
    Ok(slots)
}

/// One record's bytes.
pub(crate) fn parse_record(b: &[u8]) -> Record {
    let image = |at: usize| vk::Image::from_raw(u64_at(b, at));
    Record {
        seq: u64_at(b, 0),
        t_ms: f64::from_bits(u64_at(b, 8)),
        api: u32_at(b, 16) & API_MASK,
        end: u32_at(b, 16) & KIND_END != 0,
        handle: u32_at(b, 20),
        thread: u32_at(b, 124),
        feature: u32_at(b, 24),
        create_flags: u32_at(b, 28),
        colour: image(32),
        mvec: image(40),
        depth: image(48),
        output: image(56),
        exposure: image(64),
        jitter: [f32_at(b, 72), f32_at(b, 76)],
        mv_scale: [f32_at(b, 80), f32_at(b, 84)],
        pre_exposure: f32_at(b, 88),
        exposure_scale: f32_at(b, 92),
        reset: u32_at(b, 96) != 0,
        colour_extent: (u32_at(b, 108), u32_at(b, 112)),
        output_extent: (u32_at(b, 116), u32_at(b, 120)),
    }
}

/// A mapped ring and how far it was read.
struct Ring {
    base: *const u8,
    len: usize,
    slots: u64,
    read: u64,
}

// SAFETY: the mapping is read-only and lives as long as the process (never unmapped).
unsafe impl Send for Ring {}

impl Ring {
    /// Maps `path` read-only and removes the file (the mapping stays; the wrapper keeps its own): nothing is left in
    /// `/dev/shm` once the game exits.
    fn open(path: &std::path::Path, unlink: bool) -> Result<Ring, String> {
        use std::os::unix::io::AsRawFd;
        let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
        let len = file.metadata().map_err(|e| e.to_string())?.len() as usize;
        if len < HEADER_BYTES {
            return Err(format!("{len} bytes"));
        }
        // SAFETY: a shared read-only mapping of a file this process opened; never unmapped.
        let base = unsafe { libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ, libc::MAP_SHARED, file.as_raw_fd(), 0) };
        if base == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error().to_string());
        }
        // SAFETY: `len` bytes were just mapped.
        let bytes = unsafe { std::slice::from_raw_parts(base as *const u8, len) };
        let slots = match parse_header(bytes) {
            Ok(s) => s,
            Err(e) => {
                // SAFETY: mapped above, not used after.
                unsafe { libc::munmap(base, len) };
                return Err(e);
            }
        };
        if unlink {
            let _ = std::fs::remove_file(path);
        }
        let mut ring = Ring { base: base as *const u8, len, slots: u64::from(slots), read: 0 };
        // Only what is written from now: a ring found late (a layer loaded after the wrapper wrote) starts at its end.
        ring.read = ring.reserved();
        Ok(ring)
    }

    fn reserved(&self) -> u64 {
        // SAFETY: the header's `next` is an aligned u64 inside the mapping, written atomically by the wrapper.
        unsafe { (*(self.base.add(24) as *const AtomicU64)).load(Ordering::Acquire) }
    }

    /// The records written since the last poll, oldest first. A slot whose writer has not finished is waited for on
    /// the next poll; slots overwritten before they were read are skipped.
    fn poll(&mut self, mut each: impl FnMut(Record)) {
        let next = self.reserved();
        if next.saturating_sub(self.read) > self.slots {
            self.read = next - self.slots;
        }
        while self.read < next {
            let n = self.read;
            let at = HEADER_BYTES + (n % self.slots) as usize * RECORD_BYTES;
            debug_assert!(at + RECORD_BYTES <= self.len);
            // SAFETY: the slot lies inside the mapping; `seq` is its aligned first u64.
            let seq = unsafe { &*(self.base.add(at) as *const AtomicU64) };
            let before = seq.load(Ordering::Acquire);
            if before != n + 1 {
                if before > n + 1 {
                    // Overwritten already: skip it.
                    self.read += 1;
                    continue;
                }
                break; // still being written
            }
            let mut copy = [0u8; RECORD_BYTES];
            // SAFETY: as above; a volatile copy of memory another thread writes, checked by `seq` after.
            unsafe { std::ptr::copy_nonoverlapping(self.base.add(at), copy.as_mut_ptr(), RECORD_BYTES) };
            std::sync::atomic::fence(Ordering::Acquire);
            if seq.load(Ordering::Acquire) == before {
                each(parse_record(&copy));
            }
            self.read += 1;
        }
    }
}

/// Counters, process-wide (every device's tracker reads the one ring).
pub(crate) struct Counters {
    /// Records read.
    pub records: AtomicU64,
    /// Evaluates matched to the launch buffer that carries them ([`note_matched`]).
    pub matched: AtomicU64,
}

pub(crate) static COUNTERS: Counters = Counters { records: AtomicU64::new(0), matched: AtomicU64::new(0) };

/// The recent evaluates kept for matching.
const RECENT: usize = 64;

/// An evaluate kept for matching ([`State::recent`]).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Kept {
    record: Record,
    /// No launch belongs to it any more: its end record was read (D3D12, Vulkan), or, for D3D11, a later evaluate of its
    /// feature was taken or it was given up ([`note`]).
    ended: bool,
    /// D3D11 only: a launch buffer took it ([`pick`]); its launches in that buffer go on counting until it ends.
    taken: bool,
}

impl Kept {
    fn new(record: Record) -> Self {
        Kept { record, ended: false, taken: false }
    }
    fn open(&self) -> bool {
        !self.ended
    }
    fn d3d11(&self) -> bool {
        self.record.api == API_D3D11
    }
}

/// D3D11 evaluates not yet taken kept open per feature: DXVK's worker records an evaluate's launches a frame or two
/// after NGX's call. Older ones were skipped (the game evaluated without the launches reaching a submit) and would
/// keep every later frame matched to an older record's jitter and reset.
const D3D11_OPEN: usize = 3;

/// How far (record time) a D3D11 evaluate not yet taken may lie behind its feature's newest before it is given up:
/// several frames even at 30 fps.
const D3D11_OPEN_MS: f64 = 250.0;

struct State {
    ring: Option<Ring>,
    /// The last time the ring was looked for, while it is not open.
    looked: Option<Instant>,
    said: bool,
    /// Super Resolution and Ray Reconstruction evaluates, oldest first ([`Kept`]).
    recent: VecDeque<Kept>,
    summary_at: Option<Instant>,
}

static STATE: Mutex<State> = Mutex::new(State { ring: None, looked: None, said: false, recent: VecDeque::new(), summary_at: None });

/// How often a missing ring is looked for again (the wrapper creates it when NGX loads, after the game's device).
const LOOK_EVERY: Duration = Duration::from_secs(1);

fn configured() -> Option<std::path::PathBuf> {
    match neural_forge_protocol::env::var(RING_ENV).as_deref().map(str::trim) {
        Some("off") => None,
        Some(p) if !p.is_empty() => Some(p.into()),
        _ => Some(format!("/dev/shm/neural-forge-ngx-{}.ring", std::process::id()).into()),
    }
}

impl State {
    fn poll(&mut self) {
        if self.ring.is_none() {
            let Some(path) = configured() else { return };
            if self.looked.is_some_and(|t| t.elapsed() < LOOK_EVERY) {
                return;
            }
            self.looked = Some(Instant::now());
            if !path.exists() {
                return;
            }
            match Ring::open(&path, neural_forge_protocol::env::var(RING_ENV).is_none()) {
                Ok(ring) => {
                    crate::log!("[ngx] reading DLSS's parameters from NGX's wrapper ({})", path.display());
                    crate::logging::flush();
                    self.ring = Some(ring);
                }
                Err(e) if !self.said => {
                    self.said = true;
                    crate::log!("[ngx] {} is not usable: {e}", path.display());
                }
                Err(_) => {}
            }
        }
        let Some(ring) = self.ring.as_mut() else { return };
        let recent = &mut self.recent;
        ring.poll(|r| {
            COUNTERS.records.fetch_add(1, Ordering::Relaxed);
            note(recent, r);
        });
        if self.summary_at.is_none_or(|t| t.elapsed() >= Duration::from_secs(5)) {
            if self.summary_at.is_some() {
                summary();
            }
            self.summary_at = Some(Instant::now());
        }
    }
}

/// Keeps an upscaling evaluate in `recent`, or marks the newest open one an end record ends.
fn note(recent: &mut VecDeque<Kept>, r: Record) {
    if r.end {
        // A D3D11 evaluate's launches are recorded later, on DXVK's worker thread, after NVIDIA's call returned: its
        // end says nothing of them, so it stays open for matching (GTA San Andreas - The Definitive Edition,
        // 2026-11-11: one evaluate in thousands matched).
        if let Some(k) = recent.iter_mut().rev().find(|k| k.open() && !k.d3d11() && k.record.ended_by(&r)) {
            k.ended = true;
        }
    } else if r.is_evaluate() && r.upscales() {
        if recent.len() == RECENT {
            recent.pop_front();
        }
        recent.push_back(Kept::new(r));
        if r.api == API_D3D11 {
            give_up_skipped_d3d11(recent, &r);
        }
    }
}

/// Ends the D3D11 evaluates of `newest`'s feature that no launch took and that lie too far behind it: more than
/// [`D3D11_OPEN`] waiting, or older than [`D3D11_OPEN_MS`]. Matching takes the oldest open one, so an evaluate whose
/// launches never came would otherwise hand every later frame an older record.
fn give_up_skipped_d3d11(recent: &mut VecDeque<Kept>, newest: &Record) {
    let mut waiting = 0;
    for k in recent.iter_mut().rev() {
        if !k.open() || !k.d3d11() || k.taken || (k.record.feature, k.record.handle) != (newest.feature, newest.handle) {
            continue;
        }
        waiting += 1;
        if waiting > D3D11_OPEN || newest.t_ms - k.record.t_ms > D3D11_OPEN_MS {
            k.ended = true;
        }
    }
}

fn summary() {
    let c = &COUNTERS;
    crate::log!("[ngx] {} records, {} evaluates matched to the launches that carry them", c.records.load(Ordering::Relaxed), c.matched.load(Ordering::Relaxed));
}

fn lock() -> std::sync::MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The newest Super Resolution or Ray Reconstruction evaluate whose colour input is one of `named` (a launch's
/// images) and whose end was not read yet: NGX writes the record before it records the evaluate's launches, on the
/// same thread, and its end after the last, so a launch naming the colour input between the two is this evaluate's.
pub(crate) fn evaluate_naming(named: &[vk::Image]) -> Option<Record> {
    #[cfg(test)]
    if let Some(found) = injected(|recent| pick(&mut recent.clone(), named)) {
        return found;
    }
    let mut s = lock();
    s.poll();
    s.ring.as_ref()?;
    pick(&mut s.recent, named)
}

/// [`evaluate_naming`] over `recent`. A D3D11 evaluate's launches come later, on DXVK's worker thread, possibly after
/// the next evaluate's record: the oldest open D3D11 evaluate not yet taken that names the launch's colour input is
/// its. Taking it marks it taken, not ended: the buffer's later launches are still its (they count the writes of its
/// output); it ends when a later evaluate of its feature is taken. Each is taken once and in order.
fn pick(recent: &mut VecDeque<Kept>, named: &[vk::Image]) -> Option<Record> {
    if let Some(k) = recent.iter().rev().find(|k| k.open() && !k.d3d11() && named.contains(&k.record.colour)) {
        return Some(k.record);
    }
    let at = recent.iter().position(|k| k.open() && k.d3d11() && !k.taken && named.contains(&k.record.colour))?;
    let taken = recent[at].record;
    recent[at].taken = true;
    for k in recent.iter_mut().take(at) {
        if k.open() && k.d3d11() && (k.record.feature, k.record.handle) == (taken.feature, taken.handle) {
            k.ended = true;
        }
    }
    Some(taken)
}

/// The newest upscaling evaluate whose colour input `known` accepts (an image of the asking device): what identifies
/// DLSS's inputs there (`preupscale::Tracker::refresh`).
pub(crate) fn newest(known: impl Fn(vk::Image) -> bool) -> Option<Record> {
    #[cfg(test)]
    if let Some(found) = injected(|recent| recent.iter().rev().find(|k| known(k.record.colour)).map(|k| k.record)) {
        return found;
    }
    let mut s = lock();
    s.poll();
    s.recent.iter().rev().find(|k| known(k.record.colour)).map(|k| k.record)
}

/// Said once per process: an evaluate's colour input is not HDR.
static SAID_NOT_HDR: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// An evaluate was matched to the launch buffer that carries it: counted, and said once when NGX was told its colour
/// input is not HDR (IsHDR clear), which the encode does not treat apart.
pub(crate) fn note_matched(record: &Record) {
    COUNTERS.matched.fetch_add(1, Ordering::Relaxed);
    if !record.hdr() && !SAID_NOT_HDR.swap(true, Ordering::Relaxed) {
        crate::log!("[ngx] DLSS's colour input is not HDR (NGX's IsHDR flag is clear); the encode treats it as scene-linear all the same");
        crate::logging::flush();
    }
}

/// Whether the evaluate of record `seq` ended (its end record was read); an evaluate no longer kept ended long ago.
pub(crate) fn ended(seq: u64) -> bool {
    #[cfg(test)]
    if let Some(found) = injected(|recent| recent.iter().find(|k| k.record.seq == seq).is_none_or(|k| k.ended)) {
        return found;
    }
    let mut s = lock();
    s.poll();
    s.recent.iter().find(|k| k.record.seq == seq).is_none_or(|k| k.ended)
}

/// When DLSS Super Resolution's or Ray Reconstruction's kernels last launched in a submit with no ring open
/// ([`note_launches_without_record`]): a DLSS game whose NGX hands this process no records.
static NO_RING_LAUNCH: Mutex<Option<Instant>> = Mutex::new(None);

/// Said once per process ([`note_launches_without_record`]).
static SAID_NO_RING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether this process is a DLSS game without NGX's records: DLSS Super Resolution's or Ray Reconstruction's kernels
/// launched with no ring open, the last time less than [`crate::preupscale::PAUSE_HAND_BACK`] ago. The model stays off
/// there (nothing tells the layer DLSS's images) and the status says why ([`no_wrapper_reason`]). Frame Generation's
/// kernels do not count; a game whose DLSS stopped evaluating gets the post-upscaler path again by the timed rules.
pub(crate) fn no_wrapper() -> bool {
    #[cfg(test)]
    return NO_RING_HERE.with(|n| n.get());
    #[cfg(not(test))]
    {
        let recent = lock_no_ring().is_some_and(|t| t.elapsed() < crate::preupscale::PAUSE_HAND_BACK);
        recent && lock().ring.is_none()
    }
}

#[cfg(not(test))]
fn lock_no_ring() -> Option<Instant> {
    *NO_RING_LAUNCH.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Whether Proton was told to load Neural Forge's NGX wrapper: `WINEDLLOVERRIDES` (Proton's DLL overrides, in the
/// game's environment) gives `_nvngx` the builtin (`b`), as a Neural Forge Proton copy's `proton` script does.
pub(crate) fn wrapper_overridden(overrides: &str) -> bool {
    overrides.split(';').any(|entry| {
        let (names, mode) = entry.split_once('=').unwrap_or((entry, ""));
        mode.trim().starts_with('b') && names.split(',').any(|n| n.trim().eq_ignore_ascii_case("_nvngx"))
    })
}

/// Why DLSS's records are missing, for the status and the log: the game is on a Neural Forge Proton
/// ([`wrapper_overridden`]) but the wrapper made no ring at `path`, or it is not on one.
pub(crate) fn no_ring_text(overridden: bool, path: Option<&std::path::Path>) -> String {
    let at = path.map_or_else(|| format!("{RING_ENV}=off"), |p| p.display().to_string());
    if overridden {
        format!(
            "the game runs on a Neural Forge Proton (Proton loads its NGX wrapper) but no record of DLSS reached the layer ({at}): the wrapper did not start its ring, or something removed it. The model stays off; the wrapper's own log (NEURAL_FORGE_NGX_LOG=Z:\\tmp\\ngx.log in the launch options) says why"
        )
    } else {
        format!(
            "this game uses DLSS but is not on the Neural Forge Proton (no record of DLSS at {at}): the model stays off. Switch the game on in Neural Forge to set its Proton (Steam closed), or choose the finished frame as the placement"
        )
    }
}

/// [`no_ring_text`] for this process.
pub(crate) fn no_wrapper_reason() -> String {
    let overridden = std::env::var("WINEDLLOVERRIDES").is_ok_and(|o| wrapper_overridden(&o));
    no_ring_text(overridden, configured().as_deref())
}

/// A launch-bearing submit carried no evaluate's launches. With `dlss` (one of the launches was of DLSS Super
/// Resolution's or Ray Reconstruction's kernels, [`crate::preupscale::dlss_game::dlss_kernel_name`]) and no ring open,
/// DLSS runs without the NGX wrapper's records: the model stays off ([`no_wrapper`]), said once. Frame Generation's
/// and other CUDA launches are no sign.
pub(crate) fn note_launches_without_record(dlss: bool) {
    #[cfg(test)]
    if INJECTED.with(|i| i.borrow().is_some()) {
        return;
    }
    if !dlss {
        return;
    }
    let open = {
        let mut s = lock();
        s.poll();
        s.ring.is_some()
    };
    if open {
        return;
    }
    *NO_RING_LAUNCH.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Instant::now());
    #[cfg(test)]
    NO_RING_HERE.with(|n| n.set(true));
    if !SAID_NO_RING.swap(true, Ordering::Relaxed) {
        crate::log!("[ngx] {}", no_wrapper_reason());
        crate::logging::flush();
    }
}

/// Starts the thread that opens the ring as soon as the wrapper creates it, once per process (from the first
/// device's creation): the file is removed once mapped, so nothing else (a purge's sweep of dead games' rings) can
/// take it from the game, and the records from NGX's first evaluate on are read. Looks every 500 ms for the first 3
/// minutes, then every 5 s, until it is open.
pub(crate) fn watch() {
    static STARTED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    if configured().is_none() || STARTED.set(()).is_err() {
        return;
    }
    let started = Instant::now();
    let spawned = std::thread::Builder::new().name("nf-ngx-ring".into()).spawn(move || loop {
        {
            let mut s = lock();
            if s.ring.is_some() {
                break;
            }
            s.looked = None;
            s.poll();
            if s.ring.is_some() {
                break;
            }
        }
        std::thread::sleep(if started.elapsed() < Duration::from_secs(180) { Duration::from_millis(500) } else { Duration::from_secs(5) });
    });
    if let Err(e) = spawned {
        crate::log!("[ngx] could not start the thread that opens NGX's ring: {e}");
    }
}

/// The newest evaluate on `colour`, without counting.
pub(crate) fn latest_for(colour: vk::Image) -> Option<Record> {
    #[cfg(test)]
    if let Some(found) = injected(|recent| recent.iter().rev().find(|k| k.record.colour == colour).map(|k| k.record)) {
        return found;
    }
    let mut s = lock();
    s.poll();
    s.recent.iter().rev().find(|k| k.record.colour == colour).map(|k| k.record)
}

/// The paper white's factor for a hold on `colour` that encodes with the game's exposure as it is
/// ([`Record::paper_white_factor`]); 1 without a record.
pub(crate) fn paper_white_factor(colour: vk::Image) -> f32 {
    latest_for(colour).map_or(1.0, |r| r.paper_white_factor())
}

#[cfg(test)]
thread_local! {
    /// [`no_wrapper`] for this test's thread (the process-wide flag would leak between tests).
    static NO_RING_HERE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Records a test hands this thread's tracker instead of a ring ([`inject`]).
    static INJECTED: std::cell::RefCell<Option<Vec<Record>>> = const { std::cell::RefCell::new(None) };
}

/// Makes this thread's lookups see `records` as the ring's, in order (tests run each on its own thread).
#[cfg(test)]
pub(crate) fn inject(records: Vec<Record>) {
    INJECTED.with(|i| *i.borrow_mut() = Some(records));
}

/// `f` on the injected records as the ring would keep them ([`note`]), when a test injected any.
#[cfg(test)]
fn injected<R>(f: impl FnOnce(&VecDeque<Kept>) -> R) -> Option<R> {
    INJECTED.with(|i| {
        i.borrow().as_ref().map(|records| {
            let mut recent = VecDeque::new();
            for r in records {
                note(&mut recent, *r);
            }
            f(&recent)
        })
    })
}

/// The end record of `evaluate` ([`Record::ended_by`]), as the wrapper writes it after NVIDIA's call returned.
#[cfg(test)]
pub(crate) fn end_of(evaluate: &Record, seq: u64) -> Record {
    Record { seq, end: true, ..*evaluate }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A record's bytes as the wrapper lays them out (`crates/ngx-wrapper/src/ring.rs`, whose test checks the same
    /// offsets).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_bytes(
        seq: u64,
        feature: u32,
        flags: u32,
        images: [u64; 5],
        jitter: [f32; 2],
        mv: [f32; 2],
        pre: f32,
        scale: f32,
        reset: u32,
    ) -> [u8; RECORD_BYTES] {
        let mut b = [0u8; RECORD_BYTES];
        b[0..8].copy_from_slice(&seq.to_le_bytes());
        b[16..20].copy_from_slice(&12u32.to_le_bytes());
        b[24..28].copy_from_slice(&feature.to_le_bytes());
        b[28..32].copy_from_slice(&flags.to_le_bytes());
        for (k, i) in images.iter().enumerate() {
            b[32 + 8 * k..40 + 8 * k].copy_from_slice(&i.to_le_bytes());
        }
        for (k, v) in jitter.iter().chain(&mv).chain([&pre, &scale]).enumerate() {
            b[72 + 4 * k..76 + 4 * k].copy_from_slice(&v.to_bits().to_le_bytes());
        }
        b[96..100].copy_from_slice(&reset.to_le_bytes());
        for (k, v) in [1920u32, 1080, 1920, 1080, 3840, 2160].iter().enumerate() {
            b[100 + 4 * k..104 + 4 * k].copy_from_slice(&v.to_le_bytes());
        }
        b
    }

    fn header(slots: u32, next: u64) -> Vec<u8> {
        let mut b = vec![0u8; HEADER_BYTES + slots as usize * RECORD_BYTES];
        b[..8].copy_from_slice(&MAGIC);
        b[8..12].copy_from_slice(&VERSION.to_le_bytes());
        b[12..16].copy_from_slice(&(RECORD_BYTES as u32).to_le_bytes());
        b[16..20].copy_from_slice(&slots.to_le_bytes());
        b[24..32].copy_from_slice(&next.to_le_bytes());
        b
    }

    fn put(file: &mut [u8], slots: u32, n: u64, rec: [u8; RECORD_BYTES]) {
        let at = HEADER_BYTES + (n % u64::from(slots)) as usize * RECORD_BYTES;
        file[at..at + RECORD_BYTES].copy_from_slice(&rec);
    }

    fn img(v: u64) -> vk::Image {
        vk::Image::from_raw(v)
    }

    #[test]
    fn a_record_reads_back_field_for_field() {
        let b = record_bytes(7, FEATURE_SR, 0x4b, [0x10, 0x20, 0x30, 0x40, 0x50], [0.25, -0.5], [1920.0, 1080.0], 2.0, 0.5, 1);
        let r = parse_record(&b);
        assert_eq!(r.seq, 7);
        assert_eq!((r.api, r.end, r.feature, r.create_flags), (12, false, FEATURE_SR, 0x4b));
        assert_eq!([r.colour, r.mvec, r.depth, r.output, r.exposure], [img(0x10), img(0x20), img(0x30), img(0x40), img(0x50)]);
        assert_eq!((r.jitter, r.mv_scale, r.pre_exposure, r.exposure_scale, r.reset), ([0.25, -0.5], [1920.0, 1080.0], 2.0, 0.5, true));
        assert_eq!((r.colour_extent, r.output_extent), ((1920, 1080), (3840, 2160)));
        assert!(r.hdr() && r.upscales() && r.is_evaluate() && !r.is_rr());
        assert_eq!(r.paper_white_factor(), 4.0, "pre-exposure 2 over exposure scale 0.5");
    }

    /// An end record: the kind bit read apart from the API, and tied to its evaluate by feature, handle, thread and
    /// images; a later evaluate's end, or another thread's, does not end it.
    #[test]
    fn an_end_record_ends_its_own_evaluate() {
        let mut b = record_bytes(9, FEATURE_SR, 1, [0x10, 0x20, 0x30, 0x40, 0], [0.0; 2], [1.0; 2], 1.0, 1.0, 0);
        b[20..24].copy_from_slice(&3u32.to_le_bytes());
        b[124..128].copy_from_slice(&77u32.to_le_bytes());
        let begin = parse_record(&b);
        assert_eq!((begin.handle, begin.thread, begin.end), (3, 77, false));
        let mut e = b;
        e[0..8].copy_from_slice(&10u64.to_le_bytes());
        e[16..20].copy_from_slice(&(12 | KIND_END).to_le_bytes());
        let end = parse_record(&e);
        assert_eq!((end.api, end.end), (12, true));
        assert!(!end.is_evaluate() && begin.ended_by(&end));
        assert!(!begin.ended_by(&Record { thread: 78, ..end }), "another thread's");
        assert!(!begin.ended_by(&Record { colour: img(0x11), ..end }), "another evaluate's");
        assert!(!Record { seq: 11, ..begin }.ended_by(&end), "written before it");
        let mut recent = VecDeque::new();
        note(&mut recent, begin);
        let second = Record { seq: 11, ..begin };
        note(&mut recent, second);
        note(&mut recent, end_of(&begin, 12));
        assert_eq!(recent.iter().map(|k| (k.record.seq, k.ended)).collect::<Vec<_>>(), vec![(9, false), (11, true)], "the newest open one ends first");
    }

    /// A launch belongs to an evaluate only between its record and its end: one naming the colour input after the
    /// end (DLSS Frame Generation reading the frame) matches none.
    #[test]
    fn a_launch_matches_only_an_evaluate_that_has_not_ended() {
        let begin = parse_record(&record_bytes(1, FEATURE_SR, 1, [0x10, 0x20, 0x30, 0x40, 0], [0.0; 2], [1.0; 2], 1.0, 1.0, 0));
        inject(vec![begin]);
        assert_eq!(evaluate_naming(&[img(0x10)]).map(|r| r.seq), Some(1));
        assert!(!ended(1));
        inject(vec![begin, end_of(&begin, 2)]);
        assert_eq!(evaluate_naming(&[img(0x10)]), None);
        assert!(ended(1) && ended(5), "ended, and one not kept counts as ended");
        assert_eq!(latest_for(img(0x10)).map(|r| r.seq), Some(1), "its parameters stay readable");
    }

    #[test]
    fn the_header_is_checked() {
        assert_eq!(parse_header(&header(4, 0)), Ok(4));
        let mut wrong = header(4, 0);
        wrong[0] = b'X';
        assert!(parse_header(&wrong).is_err());
        let mut v1 = header(4, 0);
        v1[8] = 1;
        assert!(parse_header(&v1).unwrap_err().contains("version 1"), "a wrapper without end records");
        assert!(parse_header(&header(4, 0)[..HEADER_BYTES + RECORD_BYTES]).is_err(), "shorter than its slots");
    }

    #[test]
    fn unusable_exposure_values_leave_the_paper_white() {
        let r = |pre: f32, scale: f32| parse_record(&record_bytes(1, FEATURE_SR, 0, [1, 2, 3, 4, 0], [0.0; 2], [1.0; 2], pre, scale, 0)).paper_white_factor();
        assert_eq!(r(0.0, 0.0), 1.0, "not given");
        assert_eq!(r(f32::NAN, 1.0), 1.0);
        assert_eq!(r(1e9, 1.0), 1.0, "implausible");
        assert_eq!(r(0.5, 1.0), 0.5);
    }

    #[test]
    fn mv_units_need_a_plausible_scale() {
        let r = |mv: [f32; 2]| parse_record(&record_bytes(1, FEATURE_SR, 0, [1, 2, 3, 4, 0], [0.0; 2], mv, 1.0, 1.0, 0)).mv_units();
        assert_eq!(r([1920.0, 1080.0]), Some([1920.0, 1080.0]));
        assert_eq!(r([1.0, -2160.0]), Some([1.0, -2160.0]), "negative y: up-pointing motion vectors");
        assert_eq!(r([0.0, 1.0]), None);
        assert_eq!(r([f32::INFINITY, 1.0]), None);
    }

    /// A file as the wrapper leaves it: what is read, in order, and what is skipped.
    #[test]
    fn polling_reads_finished_slots_in_order() {
        let dir = std::env::temp_dir().join(format!("nf-ngx-ring-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ring");
        let slots = 4;
        let mut file = header(slots, 0);
        std::fs::write(&path, &file).unwrap();
        let mut ring = Ring::open(&path, false).unwrap();
        let mut got = Vec::new();
        ring.poll(|r| got.push(r.seq));
        assert!(got.is_empty());
        // Three records, the third still being written (seq 0).
        for n in 0..3u64 {
            put(&mut file, slots, n, record_bytes(if n == 2 { 0 } else { n + 1 }, FEATURE_SR, 0, [n + 1, 0, 0, 0, 0], [0.0; 2], [1.0; 2], 1.0, 1.0, 0));
        }
        file[24..32].copy_from_slice(&3u64.to_le_bytes());
        std::fs::write(&path, &file).unwrap();
        ring.poll(|r| got.push(r.seq));
        assert_eq!(got, vec![1, 2], "the slot being written waits");
        put(&mut file, slots, 2, record_bytes(3, FEATURE_SR, 0, [3, 0, 0, 0, 0], [0.0; 2], [1.0; 2], 1.0, 1.0, 0));
        std::fs::write(&path, &file).unwrap();
        ring.poll(|r| got.push(r.seq));
        assert_eq!(got, vec![1, 2, 3]);
        // The writer laps the reader: 10 reserved, slots hold 7..10; 4..6 are gone.
        for n in 3..10u64 {
            put(&mut file, slots, n, record_bytes(n + 1, FEATURE_SR, 0, [n + 1, 0, 0, 0, 0], [0.0; 2], [1.0; 2], 1.0, 1.0, 0));
        }
        file[24..32].copy_from_slice(&10u64.to_le_bytes());
        std::fs::write(&path, &file).unwrap();
        ring.poll(|r| got.push(r.seq));
        assert_eq!(got, vec![1, 2, 3, 7, 8, 9, 10]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_ring_opened_late_starts_at_its_end() {
        let dir = std::env::temp_dir().join(format!("nf-ngx-ring-late-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ring");
        let mut file = header(4, 2);
        put(&mut file, 4, 0, record_bytes(1, FEATURE_SR, 0, [1, 0, 0, 0, 0], [0.0; 2], [1.0; 2], 1.0, 1.0, 0));
        put(&mut file, 4, 1, record_bytes(2, FEATURE_SR, 0, [2, 0, 0, 0, 0], [0.0; 2], [1.0; 2], 1.0, 1.0, 0));
        std::fs::write(&path, &file).unwrap();
        let mut ring = Ring::open(&path, true).unwrap();
        assert!(!path.exists(), "the file is removed once mapped");
        let mut got = Vec::new();
        ring.poll(|r| got.push(r.seq));
        assert!(got.is_empty());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// DXVK records a D3D11 evaluate's launches later, after the next record may have come: the oldest open D3D11
    /// evaluate naming the colour input is the launch's, taken once and kept open (its later launches count its output
    /// writes) until a later one of its feature is taken; D3D12's newest open one, kept for its other launches.
    #[test]
    fn d3d11_evaluates_are_taken_in_order_once_each() {
        let rec = |seq: u64, api: u32| Record { seq, api, feature: FEATURE_SR, colour: vk::Image::from_raw(0x10), ..Record::default() };
        let mut recent: VecDeque<Kept> = [rec(1, API_D3D11), rec(2, API_D3D11)].into_iter().map(Kept::new).collect();
        let named = [vk::Image::from_raw(0x10)];
        assert_eq!(pick(&mut recent, &named).map(|r| r.seq), Some(1));
        assert!(!recent[0].ended && recent[0].taken, "taken, not ended: the buffer's later launches are its");
        assert_eq!(pick(&mut recent, &named).map(|r| r.seq), Some(2));
        assert!(recent[0].ended && !recent[1].ended, "the next one taken ends it");
        assert_eq!(pick(&mut recent, &named), None, "each once");
        let mut d3d12: VecDeque<Kept> = [rec(3, 12), rec(4, 12)].into_iter().map(Kept::new).collect();
        assert_eq!(pick(&mut d3d12, &named).map(|r| r.seq), Some(4));
        assert_eq!(pick(&mut d3d12, &named).map(|r| r.seq), Some(4), "kept open for the buffer's other launches");
    }

    fn d3d11(seq: u64, t_ms: f64) -> Record {
        Record { seq, t_ms, api: API_D3D11, feature: FEATURE_SR, colour: vk::Image::from_raw(0x10), ..Record::default() }
    }

    /// An evaluate whose launches never reach a submit (skipped) is given up once [`D3D11_OPEN`] newer ones wait or it
    /// lies [`D3D11_OPEN_MS`] behind: matching then takes the current frames' records again instead of staying behind.
    #[test]
    fn a_skipped_d3d11_evaluate_is_given_up() {
        let named = [vk::Image::from_raw(0x10)];
        let mut recent = VecDeque::new();
        // Frame 1's launches come; frame 2's never do; frames 3.. come one frame late.
        note(&mut recent, d3d11(1, 0.0));
        assert_eq!(pick(&mut recent, &named).map(|r| r.seq), Some(1));
        for seq in 2..=5 {
            note(&mut recent, d3d11(seq, 16.0 * seq as f64));
        }
        assert!(recent.iter().find(|k| k.record.seq == 2).unwrap().ended, "more than D3D11_OPEN waiting");
        assert_eq!(pick(&mut recent, &named).map(|r| r.seq), Some(3));
        note(&mut recent, d3d11(6, 96.0));
        assert_eq!(pick(&mut recent, &named).map(|r| r.seq), Some(4));
        // The game's frames stall for longer than D3D11_OPEN_MS: what waited is given up.
        note(&mut recent, d3d11(7, 96.0 + D3D11_OPEN_MS + 1.0));
        assert!(recent.iter().filter(|k| k.record.seq < 7).all(|k| k.ended || k.taken), "{recent:?}");
        assert_eq!(pick(&mut recent, &named).map(|r| r.seq), Some(7));
        assert!(recent.iter().filter(|k| k.record.seq < 7).all(|k| k.ended));
    }

    /// A buffer that takes an evaluate whose launches are another buffer's (the launches of one evaluate split over two
    /// buffers): the next frame's record goes early, the frame after finds none open and the matching is back in step
    /// with the frame after that.
    #[test]
    fn a_double_take_falls_back_in_step() {
        let named = [vk::Image::from_raw(0x10)];
        let mut recent = VecDeque::new();
        note(&mut recent, d3d11(1, 0.0));
        note(&mut recent, d3d11(2, 16.0));
        assert_eq!(pick(&mut recent, &named).map(|r| r.seq), Some(1));
        assert_eq!(pick(&mut recent, &named).map(|r| r.seq), Some(2), "frame 1's second buffer takes frame 2's record");
        assert_eq!(pick(&mut recent, &named), None, "frame 2's launches find none: missed, not shifted");
        note(&mut recent, d3d11(3, 32.0));
        assert_eq!(pick(&mut recent, &named).map(|r| r.seq), Some(3), "in step again");
    }

    #[test]
    fn the_wrapper_override_is_read_from_wines_dll_overrides() {
        assert!(wrapper_overridden("steam.exe=b;dotnetfx35.exe=b;_nvngx=b;Z:\\usr\\lib\\nvidia\\wine\\_nvngx=n"));
        assert!(wrapper_overridden("d3d11,dxgi,_nvngx=b,n"));
        assert!(!wrapper_overridden("steam.exe=b;nvngx=n;d3d11=n"));
        assert!(!wrapper_overridden(""));
        let on = no_ring_text(true, Some(std::path::Path::new("/dev/shm/neural-forge-ngx-42.ring")));
        let off = no_ring_text(false, Some(std::path::Path::new("/dev/shm/neural-forge-ngx-42.ring")));
        assert!(on.contains("/dev/shm/neural-forge-ngx-42.ring") && on.contains("runs on a Neural Forge Proton"), "{on}");
        assert!(off.contains("/dev/shm/neural-forge-ngx-42.ring") && off.contains("not on the Neural Forge Proton"), "{off}");
    }
}
