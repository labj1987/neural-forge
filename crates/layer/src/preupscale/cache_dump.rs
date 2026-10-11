//! The K-hold pre-upscaler cache dump: K consecutive model-mode holds written out for the edit
//! cache's offline replay (`scripts/cache_replay.py`; the format is docs/ARCHITECTURE.md,
//! "Pre-upscaler cache dump"). The model runs on every hold as always; an armed hold only adds
//! copies: the colour input before the model (in `C`), and after the write-back the colour input
//! with the answer in it, the motion vectors and the depth the hold copied (in `W`). They land in
//! host buffers built for the dump and freed after it, and are read once the hold's work finished:
//! at the next hold's `wait_idle` on the split path, right after the worker's wait inside DLSS's
//! buffer. A writer thread writes them out; the manifest is rewritten after every hold, so a dump
//! cut short is still complete up to its last hold. A hold that misses, or a DLSS submit that goes
//! unheld between two armed holds, ends the dump there.
//!
//! Requests: the channel's `preupscale_dump_request` (`shmctl dump-cache`), the trigger file next to
//! the channel (polled at most once a second) and `NEURAL_FORGE_CACHE_DUMP_AT`
//! (`neural_forge_protocol::cache_dump`).

use std::io::Write as _;
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use ash::vk;
use neural_forge_protocol::cache_dump as trigger;

use super::{own_host_buffer, HostBuffer};

/// How often the trigger file is looked for: one `Instant` compare per hold in between.
const POLL_EVERY: Duration = Duration::from_secs(1);
/// A running dump that saw no armed hold or readback for this long ends (the game stopped holding).
const STALLED: Duration = Duration::from_secs(2);

/// An armed hold: which dump it belongs to and which DLSS submit it is ([`super::Session`]'s count).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Arm {
    pub run: u64,
    pub hold: u64,
    /// The hold inside DLSS's buffer (else the split hold).
    pub inline: bool,
}

/// What an armed hold recorded, kept with the readbacks until they are read.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Pending {
    pub arm: Arm,
    pub time_ms: u64,
    pub jitter: Option<[f32; 2]>,
    /// The network used its history on this hold.
    pub history: bool,
    /// The motion vectors were copied this hold (else the file is zeros).
    pub mvec: bool,
    /// The depth the hold copied, and its format (raw: 2 bytes a texel for D16, 4 otherwise).
    pub depth: Option<vk::Format>,
    pub mvec_scale: [f32; 2],
    pub paper_white: f32,
}

/// One armed hold's bytes, read back.
pub(crate) struct Frame {
    pub meta: Pending,
    pub width: u32,
    pub height: u32,
    /// The exposure the encode and decode used.
    pub exposure: f32,
    pub colour: Vec<u8>,
    pub answer: Vec<u8>,
    pub mvec: Vec<u8>,
    /// Raw depth-aspect texels ([`Pending::depth`]); converted to f32 by the writer.
    pub depth: Option<Vec<u8>>,
}

impl std::fmt::Debug for Frame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Frame").field("meta", &self.meta).field("width", &self.width).field("height", &self.height).finish_non_exhaustive()
    }
}

/// The readbacks of one armed hold, at the render extent: colour before and after (RGBA16F), the
/// motion vectors (RG16F) and the depth (4 bytes a texel at most).
pub(crate) struct Buffers {
    width: u32,
    height: u32,
    pub(super) colour: HostBuffer,
    pub(super) answer: HostBuffer,
    pub(super) mvec: HostBuffer,
    pub(super) depth: HostBuffer,
    pub(crate) pending: Option<Pending>,
}

impl Buffers {
    /// # Safety
    /// `device` is live.
    pub(crate) unsafe fn build(device: &ash::Device, instance: &ash::Instance, physical_device: vk::PhysicalDevice, width: u32, height: u32) -> Option<Self> {
        let texels = u64::from(width) * u64::from(height);
        let mut made: Vec<HostBuffer> = Vec::new();
        for bytes in [texels * 8, texels * 8, texels * 4, texels * 4] {
            match own_host_buffer(device, instance, physical_device, bytes) {
                Some(b) => made.push(b),
                None => {
                    for b in &made {
                        // SAFETY: never submitted.
                        unsafe { b.destroy(device) };
                    }
                    return None;
                }
            }
        }
        let mut it = made.into_iter();
        let (colour, answer, mvec, depth) = (it.next()?, it.next()?, it.next()?, it.next()?);
        Some(Self { width, height, colour, answer, mvec, depth, pending: None })
    }

    /// # Safety
    /// Nothing submitted may still use the buffers.
    pub(crate) unsafe fn destroy(&self, device: &ash::Device) {
        // SAFETY: forwarded.
        unsafe {
            self.colour.destroy(device);
            self.answer.destroy(device);
            self.mvec.destroy(device);
            self.depth.destroy(device);
        }
    }

    /// The pending hold's bytes. Call only after its fence was waited on.
    pub(crate) fn take(&mut self, exposure: f32) -> Option<Frame> {
        let meta = self.pending.take()?;
        let texels = self.width as usize * self.height as usize;
        // SAFETY: each mapping covers its size, is host-coherent, and the hold's write-back fence was
        // waited on after its closing barrier made the transfers available to the host.
        let read = |b: &HostBuffer, bytes: usize| unsafe { std::slice::from_raw_parts(b.ptr, bytes) }.to_vec();
        let depth_bytes = meta.depth.map(|f| texels * super::depth_texel_bytes(f) as usize);
        Some(Frame {
            meta,
            width: self.width,
            height: self.height,
            exposure,
            colour: read(&self.colour, texels * 8),
            answer: read(&self.answer, texels * 8),
            mvec: if meta.mvec { read(&self.mvec, texels * 4) } else { vec![0; texels * 4] },
            depth: depth_bytes.map(|n| read(&self.depth, n)),
        })
    }
}

/// Raw depth-aspect texels as f32 device depth: D32 (and depth handed over as R32_SFLOAT) as they
/// are, the UNORM formats divided by their maximum (D24's top byte, X8 or stencil, dropped), and
/// R32_UINT reinterpreted as f32 bits.
pub(crate) fn depth_to_f32(format: vk::Format, raw: &[u8]) -> Vec<u8> {
    let out: Vec<f32> = match format {
        vk::Format::D16_UNORM | vk::Format::D16_UNORM_S8_UINT => raw.as_chunks::<2>().0.iter().map(|c| f32::from(u16::from_le_bytes(*c)) / 65535.0).collect(),
        vk::Format::X8_D24_UNORM_PACK32 | vk::Format::D24_UNORM_S8_UINT => {
            raw.as_chunks::<4>().0.iter().map(|c| (u32::from_le_bytes(*c) & 0x00ff_ffff) as f32 / 16_777_215.0).collect()
        }
        _ => raw.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect(),
    };
    out.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// A dump in progress.
struct Run {
    id: u64,
    want: u32,
    armed: u32,
    written: u32,
    /// The DLSS submit of the last armed hold: the next must be the one right after it.
    last_hold: Option<u64>,
    /// Armed holds whose readback has not come back yet.
    outstanding: u32,
    activity: Instant,
    dir: PathBuf,
    tx: mpsc::Sender<Frame>,
}

/// The per-device dump state ([`super::Session`]).
#[derive(Default)]
pub(crate) struct CacheDump {
    next_poll: Option<Instant>,
    at_taken: bool,
    runs: u64,
    run: Option<Run>,
}

impl CacheDump {
    /// Takes a pending request (the channel's field: one atomic load; the trigger file: at most once
    /// a second; `NEURAL_FORGE_CACHE_DUMP_AT`: against the session's resumes) and starts a dump.
    /// A request while a dump runs is taken and dropped.
    pub(crate) fn poll(&mut self, shm: &crate::shm::ShmClient, resumes: u32, resumed_at: Option<Instant>) {
        let now = Instant::now();
        if self.run.as_ref().is_some_and(|r| now.duration_since(r.activity) > STALLED) {
            self.end("no hold for 2 s");
        }
        let mut want = shm.take_preupscale_dump_request();
        if self.next_poll.is_none_or(|t| now >= t) {
            self.next_poll = Some(now + POLL_EVERY);
            if !shm.channel_path().is_empty() {
                want = want.or_else(|| take_trigger_file(&trigger::trigger_path(shm.channel_path())));
            }
        }
        if !self.at_taken {
            if let Some((n, after, k)) = scheduled() {
                if resumes == n && resumed_at.is_some_and(|t| t.elapsed() >= after) {
                    self.at_taken = true;
                    crate::log!("[preupscale] NEURAL_FORGE_CACHE_DUMP_AT: dumping {k} holds {} ms after resume {n}", after.as_millis());
                    want = want.or(Some(k));
                }
            }
        }
        let Some(k) = want else { return };
        if self.run.is_some() {
            crate::log!("[preupscale] cache dump of {k} holds requested while one runs; ignored");
            return;
        }
        let k = k.clamp(1, trigger::MAX_HOLDS);
        self.runs += 1;
        let stamp = unix_ms();
        let dir = crate::dump::captures_dir().join(format!("preupscale-cache-{stamp}"));
        let (tx, rx) = mpsc::channel::<Frame>();
        let writer_dir = dir.clone();
        let spawned = std::thread::Builder::new().name("nf-cache-dump".into()).spawn(move || write_frames(&writer_dir, &rx));
        if spawned.is_err() {
            crate::log!("[preupscale] could not start the cache dump writer thread");
            return;
        }
        crate::log!("[preupscale] cache dump of {k} model-mode holds to {}", dir.display());
        crate::logging::flush();
        self.run = Some(Run { id: self.runs, want: k, armed: 0, written: 0, last_hold: None, outstanding: 0, activity: now, dir, tx });
    }

    /// Whether the hold for DLSS submit `hold` should be armed.
    pub(crate) fn arm(&self, hold: u64, inline: bool) -> Option<Arm> {
        let run = self.run.as_ref()?;
        (run.armed < run.want).then_some(Arm { run: run.id, hold, inline })
    }

    /// Whether a dump runs (the split hold refuses to arm one on a colour input it cannot copy).
    pub(crate) fn running(&self) -> bool {
        self.run.is_some()
    }

    /// Books a hold's dump part: a readback it brought (written), then whether the hold armed for
    /// the dump (`arm`) did its job (`wrote_back`); a miss or a gap ends the dump.
    pub(crate) fn after_hold(&mut self, arm: Option<Arm>, wrote_back: bool, frame: Option<Frame>) {
        if let Some(frame) = frame {
            self.received(frame);
        }
        let Some(arm) = arm else { return };
        let Some(run) = self.run.as_mut().filter(|r| r.id == arm.run) else { return };
        if !wrote_back {
            self.end("an armed hold missed");
            return;
        }
        if run.last_hold.is_some_and(|last| arm.hold != last + 1) {
            self.end("a DLSS frame between two armed holds went unheld");
            return;
        }
        run.last_hold = Some(arm.hold);
        run.armed += 1;
        run.outstanding += 1;
        run.activity = Instant::now();
    }

    /// A pending hold that will not be read back (its resources were rebuilt, or it was dropped).
    pub(crate) fn lost(&mut self) {
        if self.run.as_ref().is_some_and(|r| r.outstanding > 0) {
            self.end("an armed hold's readback was lost");
        }
    }

    fn received(&mut self, frame: Frame) {
        let Some(run) = self.run.as_mut().filter(|r| r.id == frame.meta.arm.run) else { return };
        run.outstanding = run.outstanding.saturating_sub(1);
        run.written += 1;
        run.activity = Instant::now();
        if run.tx.send(frame).is_err() {
            self.end("the writer thread ended");
            return;
        }
        if run.written >= run.want {
            self.end("complete");
        }
    }

    /// Ends a running dump (what is written stays, with its manifest).
    pub(crate) fn end(&mut self, why: &str) {
        if let Some(run) = self.run.take() {
            crate::log!("[preupscale] cache dump {} ended ({why}): {} of {} holds, written to {}", run.id, run.written, run.want, run.dir.display());
            crate::logging::flush();
        }
    }
}

/// `NEURAL_FORGE_CACHE_DUMP_AT`, parsed once.
fn scheduled() -> Option<(u32, Duration, u32)> {
    static AT: std::sync::LazyLock<Option<(u32, Duration, u32)>> =
        std::sync::LazyLock::new(|| neural_forge_protocol::env::var("NEURAL_FORGE_CACHE_DUMP_AT").and_then(|v| trigger::parse_at(&v)));
    *AT
}

/// Reads and removes the trigger file, if there is one. A file that is not a request is removed
/// too (and said why).
pub(crate) fn take_trigger_file(path: &std::path::Path) -> Option<u32> {
    let text = std::fs::read_to_string(path).ok()?;
    let _ = std::fs::remove_file(path);
    let k = trigger::parse_trigger(&text);
    if k.is_none() {
        crate::log!("[preupscale] {}: not a hold count from 1 to {} ({:?}); removed", path.display(), trigger::MAX_HOLDS, text.trim());
    }
    k
}

fn unix_ms() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64)
}

/// A JSON number, or `null` for a value JSON cannot hold.
fn number(v: f32) -> String {
    if v.is_finite() { format!("{v}") } else { "null".into() }
}

/// One frame's manifest entry.
fn entry(index: u32, frame: &Frame, files: &[(&str, String)]) -> String {
    let jitter = frame.meta.jitter.map_or_else(|| "null".to_string(), |[x, y]| format!("[{}, {}]", number(x), number(y)));
    let files: Vec<String> = files.iter().map(|(k, v)| format!("\"{k}\": \"{v}\"")).collect();
    format!(
        "    {{\"index\": {index}, \"hold\": {}, \"time_ms\": {}, \"jitter\": {jitter}, \"exposure\": {}, \"history\": {}, \"mvec_valid\": {}, \"depth_valid\": {}, \"files\": {{{}}}}}",
        frame.meta.arm.hold,
        frame.meta.time_ms,
        number(frame.exposure),
        frame.meta.history,
        frame.meta.mvec,
        frame.meta.depth.is_some(),
        files.join(", ")
    )
}

/// The manifest for the frames written so far (`first`: the first frame, for the dump's constants).
fn manifest(first: &Frame, depth_format: Option<vk::Format>, entries: &[String]) -> String {
    format!(
        "{{\n  \"format\": \"neural-forge-cache-dump\", \"version\": 1,\n  \"holds\": {},\n  \"width\": {}, \"height\": {},\n  \"path\": \"{}\",\n  \
         \"paper_white\": {},\n  \"mvec_scale\": [{}, {}],\n  \"depth_format\": \"{}\",\n  \"reverse_z\": null,\n  \"game\": \"{}\",\n  \"frames\": [\n{}\n  ]\n}}\n",
        entries.len(),
        first.width,
        first.height,
        if first.meta.arm.inline { "inline" } else { "split" },
        number(first.meta.paper_white),
        number(first.meta.mvec_scale[0]),
        number(first.meta.mvec_scale[1]),
        depth_format.map_or_else(|| "none".to_string(), |f| format!("{f:?}")),
        crate::ownership::process_name().replace(['"', '\\'], "_"),
        entries.join(",\n")
    )
}

/// Writes one frame's files into `dir`; returns its manifest entry.
fn write_frame(dir: &std::path::Path, index: u32, frame: &Frame) -> std::io::Result<String> {
    let sub = format!("hold-{index:02}");
    std::fs::create_dir_all(dir.join(&sub))?;
    let mut files = vec![("colour", format!("{sub}/colour.rgba16f")), ("answer", format!("{sub}/answer.rgba16f")), ("mvec", format!("{sub}/mvec.rg16f"))];
    std::fs::write(dir.join(&files[0].1), &frame.colour)?;
    std::fs::write(dir.join(&files[1].1), &frame.answer)?;
    std::fs::write(dir.join(&files[2].1), &frame.mvec)?;
    if let (Some(raw), Some(format)) = (&frame.depth, frame.meta.depth) {
        let name = format!("{sub}/depth.r32f");
        std::fs::write(dir.join(&name), depth_to_f32(format, raw))?;
        files.push(("depth", name));
    }
    Ok(entry(index, frame, &files))
}

/// The writer thread: every frame as it comes, the manifest rewritten after each (through a
/// temporary file, so a reader never sees half of one).
pub(crate) fn write_frames(dir: &std::path::Path, rx: &mpsc::Receiver<Frame>) {
    let mut entries = Vec::new();
    let mut first: Option<Frame> = None;
    let mut depth_format = None;
    for frame in rx {
        let index = entries.len() as u32;
        match write_frame(dir, index, &frame) {
            Ok(e) => entries.push(e),
            Err(e) => {
                crate::log!("[preupscale] cache dump to {} failed at hold {index}: {e}", dir.display());
                crate::logging::flush();
                return;
            }
        }
        depth_format = depth_format.or(frame.meta.depth);
        // The constants come from the first frame; only its metadata is kept.
        let head = first.get_or_insert(Frame { colour: Vec::new(), answer: Vec::new(), mvec: Vec::new(), depth: None, ..frame });
        let text = manifest(head, depth_format, &entries);
        let tmp = dir.join("manifest.json.tmp");
        let written = std::fs::File::create(&tmp).and_then(|mut f| f.write_all(text.as_bytes())).and_then(|()| std::fs::rename(&tmp, dir.join("manifest.json")));
        if let Err(e) = written {
            crate::log!("[preupscale] cache dump manifest in {} failed: {e}", dir.display());
            crate::logging::flush();
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(hold: u64, depth: Option<vk::Format>) -> Pending {
        Pending {
            arm: Arm { run: 1, hold, inline: false },
            time_ms: 1_791_680_000_000 + hold,
            jitter: Some([0.25, -0.125]),
            history: hold > 10,
            mvec: true,
            depth,
            mvec_scale: [1.0, 1.0],
            paper_white: 3.0,
        }
    }

    /// A normal f32 as a half (exact for the test's values).
    fn half(v: f32) -> [u8; 2] {
        let b = v.to_bits();
        let exp = ((b >> 23) & 0xff) + 15 - 127;
        (((b >> 16) & 0x8000) as u16 | (exp << 10) as u16 | ((b >> 13) & 0x3ff) as u16).to_le_bytes()
    }

    /// A tiny frame: colour 0.5 grey, the answer twice it, vectors (1, -0.5), depth 0.25.
    fn frame(hold: u64, w: u32, h: u32, depth: Option<vk::Format>) -> Frame {
        let n = (w * h) as usize;
        let rgba = |v: f32| (0..n).flat_map(|_| [half(v), half(v), half(v), half(1.0)].concat()).collect::<Vec<u8>>();
        Frame {
            meta: meta(hold, depth),
            width: w,
            height: h,
            exposure: 0.2,
            colour: rgba(0.5),
            answer: rgba(1.0),
            mvec: (0..n).flat_map(|_| [half(1.0), half(-0.5)].concat()).collect(),
            depth: depth.map(|_| (0..n).flat_map(|_| 0.25f32.to_le_bytes()).collect()),
        }
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(2).unwrap().join("target/test-scratch").join(format!("cache-dump-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn depth_is_written_as_f32_device_depth() {
        let d16 = depth_to_f32(vk::Format::D16_UNORM, &[0xff, 0xff, 0, 0]);
        assert_eq!(d16, [1.0f32.to_le_bytes(), 0.0f32.to_le_bytes()].concat());
        let d24 = depth_to_f32(vk::Format::D24_UNORM_S8_UINT, &0xab_ff_ff_ffu32.to_le_bytes());
        assert_eq!(d24, 1.0f32.to_le_bytes());
        let d32 = depth_to_f32(vk::Format::D32_SFLOAT_S8_UINT, &0.375f32.to_le_bytes());
        assert_eq!(d32, 0.375f32.to_le_bytes());
    }

    /// The writer's files and manifest are what `scripts/cache_common.py` reads: the sizes and fields
    /// here, and, where python3 with numpy is installed, the loader itself.
    #[test]
    fn the_writer_writes_what_the_replay_tool_reads() {
        let dir = scratch("writer");
        let (w, h) = (6, 4);
        let (tx, rx) = mpsc::channel();
        for hold in 10..13 {
            tx.send(frame(hold, w, h, Some(vk::Format::D32_SFLOAT_S8_UINT))).unwrap();
        }
        drop(tx);
        write_frames(&dir, &rx);
        let text = std::fs::read_to_string(dir.join("manifest.json")).unwrap();
        for want in ["\"holds\": 3", "\"width\": 6, \"height\": 4", "\"path\": \"split\"", "\"depth_format\": \"D32_SFLOAT_S8_UINT\"", "\"hold\": 12", "\"jitter\": [0.25, -0.125]"] {
            assert!(text.contains(want), "{want} missing from {text}");
        }
        let size = |name: &str| std::fs::metadata(dir.join(name)).unwrap().len();
        assert_eq!(size("hold-02/colour.rgba16f"), 6 * 4 * 8);
        assert_eq!(size("hold-02/answer.rgba16f"), 6 * 4 * 8);
        assert_eq!(size("hold-02/mvec.rg16f"), 6 * 4 * 4);
        assert_eq!(size("hold-02/depth.r32f"), 6 * 4 * 4);
        assert!(!dir.join("manifest.json.tmp").exists());
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(2).unwrap();
        let check = format!(
            "import sys; sys.path.insert(0, {:?}); import cache_common as cc; d = cc.load_dump({:?}); \
             assert len(d) == 3 and d.has_depth and (d.width, d.height) == (6, 4); h = d.hold(2); \
             assert abs(float(h.colour[0, 0, 0]) - 0.5) < 1e-3 and abs(float(h.answer[3, 5, 1]) - 1.0) < 1e-3; \
             assert abs(float(h.mvec[1, 1, 0]) - 1.0) < 1e-3 and abs(float(h.mvec[1, 1, 1]) + 0.5) < 1e-3; \
             assert abs(float(h.depth[2, 2]) - 0.25) < 1e-6 and list(h.jitter) == [0.25, -0.125]; print('ok')",
            root.join("scripts").display().to_string(),
            dir.display().to_string()
        );
        match std::process::Command::new("python3").args(["-I", "-c", "import numpy"]).status() {
            Ok(s) if s.success() => {
                let out = std::process::Command::new("python3").args(["-I", "-B", "-c", &check]).output().unwrap();
                assert!(out.status.success(), "the replay tool's loader refused the dump: {}", String::from_utf8_lossy(&out.stderr));
            }
            _ => eprintln!("python3 with numpy not found: the loader check is skipped"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_dump_without_depth_says_none_and_writes_no_depth_files() {
        let dir = scratch("nodepth");
        let (tx, rx) = mpsc::channel();
        tx.send(frame(1, 2, 2, None)).unwrap();
        drop(tx);
        write_frames(&dir, &rx);
        let text = std::fs::read_to_string(dir.join("manifest.json")).unwrap();
        assert!(text.contains("\"depth_format\": \"none\""), "{text}");
        assert!(!dir.join("hold-00/depth.r32f").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_trigger_file_is_taken_once() {
        let dir = scratch("trigger");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(trigger::TRIGGER_FILE);
        std::fs::write(&path, "8\n").unwrap();
        assert_eq!(take_trigger_file(&path), Some(8));
        assert!(!path.exists(), "the layer removes the file it took");
        assert_eq!(take_trigger_file(&path), None);
        std::fs::write(&path, "").unwrap();
        assert_eq!(take_trigger_file(&path), Some(trigger::DEFAULT_HOLDS));
        std::fs::write(&path, "nonsense").unwrap();
        assert_eq!(take_trigger_file(&path), None);
        assert!(!path.exists(), "a file that is no request is removed too");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Consecutive armed holds are written; a gap between them, or a miss, ends the dump.
    #[test]
    fn a_gap_or_a_miss_ends_the_dump() {
        let (tx, rx) = mpsc::channel();
        let mut d = CacheDump { run: Some(Run { id: 1, want: 4, armed: 0, written: 0, last_hold: None, outstanding: 0, activity: Instant::now(), dir: PathBuf::new(), tx }), runs: 1, ..Default::default() };
        let a = d.arm(10, false).unwrap();
        d.after_hold(Some(a), true, None);
        let b = d.arm(11, false).unwrap();
        d.after_hold(Some(b), true, Some(frame(10, 1, 1, None)));
        assert!(d.running());
        // DLSS submit 12 went unheld: 13 does not follow 11.
        let c = d.arm(13, false).unwrap();
        d.after_hold(Some(c), true, Some(frame(11, 1, 1, None)));
        assert!(!d.running());
        assert_eq!(rx.try_iter().map(|f| f.meta.arm.hold).collect::<Vec<_>>(), [10, 11]);
        let (tx, _rx) = mpsc::channel();
        d.run = Some(Run { id: 2, want: 4, armed: 0, written: 0, last_hold: None, outstanding: 0, activity: Instant::now(), dir: PathBuf::new(), tx });
        let a = d.arm(20, false).unwrap();
        d.after_hold(Some(a), false, None);
        assert!(!d.running(), "a missed armed hold ends the dump");
    }

    #[test]
    fn the_dump_ends_after_k_holds() {
        let (tx, rx) = mpsc::channel();
        let mut d = CacheDump { run: Some(Run { id: 1, want: 2, armed: 0, written: 0, last_hold: None, outstanding: 0, activity: Instant::now(), dir: PathBuf::new(), tx }), runs: 1, ..Default::default() };
        let a = d.arm(1, false).unwrap();
        d.after_hold(Some(a), true, None);
        let b = d.arm(2, false).unwrap();
        d.after_hold(Some(b), true, Some(frame(1, 1, 1, None)));
        assert!(d.arm(3, false).is_none(), "K holds armed");
        d.after_hold(None, true, Some(frame(2, 1, 1, None)));
        assert!(!d.running());
        assert_eq!(rx.try_iter().count(), 2);
    }
}
