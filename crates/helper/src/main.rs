//! `neural-forge-helper.exe` — Windows-side NGX service.
//!
//! Owns its own Vulkan device and the `nvngx_dlssnr.dll` model, waits on the
//! shared-memory frame queue the layer writes to, runs the neural pass, and returns
//! processed frames. Built for `x86_64-pc-windows-gnu` and run under Wine/Proton —
//! see `CLAUDE.md` for what has and hasn't actually been verified on this dev machine.
//!
//! Milestone 4: watches `seq_req` for a change, ensures the NGX feature exists at
//! that frame's real size (`ngx::maintain_feature`, deferred from startup since there's
//! no real size before the layer's first capture), runs `EvaluateFeature` through
//! `frame::FrameResources`, and writes the result into the answer region -- or, if the
//! feature isn't ready (still building, or the guarded `CreateFeature`/
//! `EvaluateFeature` call failed), echoes the proxy bytes straight through so the
//! transport itself still proves out even when the model side doesn't.
//!
//! `frame_resources` is rebuilt whenever the observed width/height changes (e.g. two
//! separate Vulkan processes -- the game and the Steam overlay -- both driving this
//! same helper before `swapchain::is_plausible_game_size` existed to filter the
//! overlay's own swapchain out on the layer side) -- the *old* `FrameResources` goes
//! through `retire_frame_resources` first, which explicitly destroys it via
//! `FrameResources::destroy` (letting the `Option` just get overwritten would leak its
//! images/memory/command pool every time, not free them -- `ash` handles are not
//! `Drop`) unless its own bounded fence wait already timed out, in which case it is
//! deliberately leaked instead of destroyed (see `FrameResources::stalled`'s doc
//! comment for why).
//!
//! This is a thin wrapper around the `neural_forge_helper` library crate (see `lib.rs`) --
//! that split exists so `examples/` can exercise individual modules directly.

// Suppresses the console window Wine/Windows would otherwise pop up for this
// process -- a plain Rust binary links as a CONSOLE-subsystem PE by default, and
// this helper never has anything to print to one that matters: real deployments
// always set `NEURAL_FORGE_LOG` (`neural_forge_supervisor::start()`, confirmed by grep), so
// `crate::logging`'s own `Stderr` fallback is already unreachable in practice --
// see that module's own doc comment. Found real, reported by the user, 2026-09-11:
// this window shows up on every real launch and does nothing (no input, no output
// worth reading), purely a side effect of never having set this.
#![windows_subsystem = "windows"]

use std::sync::atomic::Ordering;
use std::time::Duration;

use ash::vk;
use neural_forge_helper::{frame, guard, ngx, optical_flow, shm};

/// The helper's motion-vector state across frames: the GPU flow session (rebuilt when the
/// model's frame size or the quality changes), a small luma thumbnail of the last frame for
/// scene-cut detection, and a latch that stops retrying after a failure until the toggle is
/// switched off and on again.
#[derive(Default)]
struct MotionState {
    flow: Option<optical_flow::GpuFlow>,
    thumb: Vec<u8>,
    blocked: bool,
    frames: u64,
}

impl MotionState {
    /// Per frame, before evaluating: drops everything when motion is off (an explicit off
    /// state, freeing the session), otherwise updates the thumbnail and returns whether this
    /// frame is a scene cut.
    fn prepare(&mut self, device: &ash::Device, want: bool, proxy: &[u8], width: u32, height: u32) -> bool {
        if !want {
            if let Some(f) = self.flow.take() {
                // SAFETY: every `estimate` waits for its own work before returning.
                unsafe { f.destroy(device) };
            }
            self.thumb.clear();
            self.blocked = false;
            return false;
        }
        let thumb = optical_flow::luma_thumbnail(proxy, width, height);
        let cut = optical_flow::is_scene_cut(&self.thumb, &thumb, 40);
        self.thumb = thumb;
        cut
    }

    /// The flow session for this frame, (re)built as needed, or `None` when motion can't run.
    #[allow(clippy::too_many_arguments)]
    fn session(&mut self, instance: &ash::Instance, device: &ash::Device, physical_device: vk::PhysicalDevice, flow_queue: Option<&optical_flow::FlowQueue>, width: u32, height: u32, quality: u32, scene_cut: bool) -> Option<&mut optical_flow::GpuFlow> {
        if self.blocked {
            return None;
        }
        let Some(flow_queue) = flow_queue else {
            static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !LOGGED.swap(true, Ordering::Relaxed) {
                neural_forge_helper::log!("[mvec] motion vectors are on but this helper has no optical-flow queue; they take effect the next time the helper starts");
            }
            return None;
        };
        if !self.flow.as_ref().is_some_and(|f| (f.width, f.height, f.quality) == (width, height, quality)) {
            if let Some(old) = self.flow.take() {
                // SAFETY: every `estimate` waits for its own work before returning.
                unsafe { old.destroy(device) };
            }
            // Family 0: the model's images and main queue (`FrameResources::new(.., 0, ..)`).
            match optical_flow::GpuFlow::new(instance, device, physical_device, 0, flow_queue, width, height, quality) {
                Ok(f) => self.flow = Some(f),
                Err(e) => {
                    neural_forge_helper::log!("[mvec] optical flow session unavailable at {width}x{height}: {e}");
                    self.blocked = true;
                    return None;
                }
            }
        }
        let flow = self.flow.as_mut()?;
        if scene_cut {
            neural_forge_helper::log!("[mvec] scene cut detected, resetting motion history");
            flow.reset();
        }
        Some(flow)
    }

    /// After evaluating: drops a failed session (and stops retrying), and logs the estimate
    /// time now and then.
    fn finished(&mut self, device: &ash::Device, timing: &frame::FrameTiming) {
        if timing.motion_failed {
            if let Some(f) = self.flow.take() {
                // SAFETY: a failed `estimate` either waited for its work or marked itself
                // stalled, in which case `destroy` leaks instead.
                unsafe { f.destroy(device) };
            }
            self.blocked = true;
        }
        if let Some(t) = timing.motion {
            self.frames += 1;
            if self.frames % 300 == 1 {
                neural_forge_helper::log!("[mvec] estimate {:.2} ms", t.as_secs_f64() * 1000.0);
            }
        }
    }
}

fn store_ms(field: &std::sync::atomic::AtomicU32, duration: Duration) {
    field.store(
        ((duration.as_secs_f64() * 1_000.0) as f32).to_bits(),
        Ordering::Relaxed,
    );
}

fn main() {
    guard::install();

    // Test-only: artificially delays every response by this many milliseconds,
    // simulating a helper that genuinely takes far longer than one frame to answer.
    // Read once at startup (this never needs to change mid-run) so the per-frame loop
    // below pays nothing but a single `Duration` comparison when it's unset -- the
    // default, real, deployed case. See `docs/ASYNC_CAPTURE_DESIGN.md`'s own validation
    // section and docs/PHASE1.md's Phase 2 item 6 for why this exists: proving the layer's
    // present hook never blocks needs a helper slow enough that blocking would be
    // obvious, not just "usually fast".
    let helper_delay: Duration = neural_forge_protocol::env::var("NEURAL_FORGE_HELPER_DELAY_MS")
        .and_then(|s| s.parse::<u64>().ok())
        .map(Duration::from_millis)
        .unwrap_or_default();

    let Some(shm) = shm::open() else {
        neural_forge_helper::log!("[helper] failed to open the shared-memory mapping");
        return;
    };
    // SAFETY: `shm.header` was just validated by `shm::open`.
    let hdr = unsafe { &*shm.header };
    hdr.helper_state.store(neural_forge_protocol::enums::helper_state::STARTING, Ordering::Relaxed);
    hdr.control_seq.fetch_add(1, Ordering::Relaxed);
    hdr.heartbeat.fetch_add(1, Ordering::Relaxed);
    neural_forge_helper::log!("[helper] shm attached");
    neural_forge_helper::logging::flush();

    let Some((entry, instance, physical_device, device, queue, flow_queue, flow_status)) = create_vulkan_context(hdr.mvec_enabled()) else {
        neural_forge_helper::log!("[helper] failed to create a Vulkan context");
        neural_forge_helper::logging::flush();
        hdr.helper_state.store(neural_forge_protocol::enums::helper_state::NO_VULKAN, Ordering::Relaxed);
        // SAFETY: nothing else references `shm` after this; it owns its own handles.
        unsafe { shm.close() };
        return;
    };
    neural_forge_helper::log!("[helper] Vulkan context created, loading NGX next");
    neural_forge_helper::logging::flush();

    let mut snippet = ngx::load_and_init(instance.handle(), physical_device, device.handle());
    hdr.helper_state.store(
        if snippet.disabled {
            neural_forge_protocol::enums::helper_state::MODEL_FAILED
        } else {
            neural_forge_protocol::enums::helper_state::RUNNING
        },
        Ordering::Relaxed,
    );
    neural_forge_helper::log!("[helper] NGX snippet disabled={}", snippet.disabled);
    neural_forge_helper::log!("[mvec] optical flow queue: {flow_status}");
    neural_forge_helper::logging::flush();

    // Protocol v3 (`docs/PROTOCOL_V3_DESIGN.md`): one persistent `FrameResources` per wire
    // slot, each importing (or staging into) that slot's own disjoint proxy/answer
    // region -- so slot 1's upload never has to wait on slot 0's own resources being
    // free. `snippet` (the single NGX feature/model) is deliberately *not*
    // duplicated -- shared across both calls to `process_request` below, since this
    // project has no evidence a single reverse-engineered NGX feature handle is safe
    // to evaluate from two overlapping submissions (see that design doc's own
    // reasoning). Each slot is still processed to completion, one at a time, before
    // the other is even checked this same loop tick.
    let mut frame_resources: [Option<frame::FrameResources>; 2] = [None, None];
    // Slot 0 only: slot 1 always evaluates with empty motion (see `process_request`).
    let mut motion = MotionState::default();
    // Resized (not reallocated fresh every frame) to whatever the current frame's
    // real byte count is -- never the full `MAX_FRAME` reservation, which is sized for
    // the protocol's absolute ceiling (7680x4320 float16), not a typical frame.
    let mut last_seq_req = [hdr.seq_req.load(Ordering::Acquire), hdr.seq_req_b.load(Ordering::Acquire)];
    let mut frames: u64 = 0;

    loop {
        if hdr.quit.load(Ordering::Relaxed) != 0 {
            break;
        }
        // Restated every iteration, not once at startup: the layer or the GUI re-initialising the
        // header resets `helper_state` to zero, and a state stated only once stays wrong (reading
        // as stopped) for the rest of the session.
        hdr.helper_state.store(
            if snippet.disabled {
                neural_forge_protocol::enums::helper_state::MODEL_FAILED
            } else {
                neural_forge_protocol::enums::helper_state::RUNNING
            },
            Ordering::Relaxed,
        );
        for slot in 0..2 {
            let seq_req = hdr.seq_req_slot(slot).load(Ordering::Acquire);
            if seq_req == last_seq_req[slot] {
                continue;
            }
            // A value behind the last one seen means the header was reinitialised under us
            // (wrapping distance, so a genuine u32 wrap is not mistaken for it). Whatever the
            // feature was built against is stale: drop it and let the next frame rebuild.
            if seq_req.wrapping_sub(last_seq_req[slot]) > u32::MAX / 2 {
                neural_forge_helper::log!(
                    "[helper] slot {slot}: seq_req went backwards ({} -> {seq_req}); header reinitialised, resetting the feature",
                    last_seq_req[slot]
                );
                ngx::discard_features(&mut snippet, &device);
            }
            last_seq_req[slot] = seq_req;
            process_request(
                hdr, &shm, &device, &instance, physical_device, queue, &mut snippet,
                &mut frame_resources[slot], flow_queue.as_ref(), &mut motion,
                slot, seq_req, helper_delay, &mut frames,
            );
        }
        hdr.heartbeat.fetch_add(1, Ordering::Relaxed);
        std::thread::sleep(Duration::from_micros(200));
    }

    // SAFETY: process is tearing down; nothing else can still be submitting work
    // against `frame_resources`'s handles.
    for f in frame_resources {
        if let Some(f) = f {
            unsafe { f.destroy(&device) };
        }
    }
    // SAFETY: same reasoning -- `estimate`'s own fence wait already drained
    // whatever this session last submitted, and the loop above just stopped.
    if let Some(f) = motion.flow {
        unsafe { f.destroy(&device) };
    }
    ngx::teardown(snippet);
    hdr.helper_state.store(neural_forge_protocol::enums::helper_state::STOPPED, Ordering::Relaxed);
    // SAFETY: destroyed in the reverse order of creation; nothing else holds a
    // reference to `device`/`instance` past this point.
    unsafe {
        device.destroy_device(None);
        instance.destroy_instance(None);
        shm.close();
    }
    drop(entry);
}

/// Destroys `old`, unless a bounded fence wait on it already timed out (`stalled`) --
/// see `frame::FrameResources::stalled`'s own doc comment for why that case must leak
/// instead: the GPU work the timeout gave up waiting on may still be running, and
/// freeing images/memory/a command pool out from under it would be unsound. A leak
/// here is a one-time, anomalous cost in a process that will usually just keep running
/// at the new size on a freshly built instance, not a routine one -- logged once so a
/// real occurrence is visible without repeating on every later resize.
fn retire_frame_resources(old: frame::FrameResources, device: &ash::Device) {
    if old.stalled() {
        static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !LOGGED.swap(true, Ordering::Relaxed) {
            neural_forge_helper::log!("[helper] leaking a stalled frame-resource instance instead of destroying it (its own fence wait never returned)");
        }
        return;
    }
    // SAFETY: not stalled, so `FrameResources::evaluate` waited on its own fences
    // (bounded, but successfully) before returning every time it was called -- nothing
    // is still submitted.
    unsafe { old.destroy(device) };
}

/// Handles one newly-observed request on the given wire slot: reads its width/
/// height/proxy_format (via the header's own `*_slot` accessors -- see
/// `docs/PROTOCOL_V3_DESIGN.md`), prewarms or evaluates against `frame_resources` (that
/// slot's own, independent from the other slot's), and publishes the answer plus
/// this slot's `seq_resp`. Exactly the per-request body `main`'s loop used to run
/// inline for the single slot v2 had; pulled out so both slots run the identical
/// logic instead of a copy that could drift, not because either slot is special.
///
/// Motion vectors are estimated for slot 0 only (the optical-flow session holds one
/// history), so slot 1 always evaluates with empty motion, same as slot 0 does
/// whenever motion is off. `hdr.seq_ok` is likewise shared, not per-slot: nothing
/// anywhere in this workspace ever reads it back (confirmed by grep), so there is
/// nothing to race by having both slots write the same dead field.
#[allow(clippy::too_many_arguments)]
fn process_request(
    hdr: &neural_forge_protocol::ShmHeader,
    shm: &shm::ShmMapping,
    device: &ash::Device,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    queue: vk::Queue,
    snippet: &mut ngx::NgxSnippet,
    frame_resources: &mut Option<frame::FrameResources>,
    flow_queue: Option<&optical_flow::FlowQueue>,
    motion: &mut MotionState,
    slot: usize,
    seq_req: u32,
    helper_delay: Duration,
    frames: &mut u64,
) {
    let width = hdr.width_slot(slot).load(Ordering::Relaxed);
    let height = hdr.height_slot(slot).load(Ordering::Relaxed);
    let proxy_format = hdr.proxy_format_slot(slot).load(Ordering::Relaxed);
    let bytes = neural_forge_protocol::enums::proxy_format::bytes_per_pixel(proxy_format)
        .saturating_mul(width as usize)
        .saturating_mul(height as usize)
        .min(neural_forge_protocol::MAX_FRAME);
    // These three values come from shared memory another process writes -- never
    // size a Vulkan image from them unchecked. A rejected frame still completes its
    // round trip (fail-open echo below), it just never reaches `FrameResources::new`,
    // NGX, or optical flow.
    let dims_ok = neural_forge_protocol::frame_dims_valid(width, height, proxy_format);
    if !dims_ok {
        neural_forge_helper::log!("[helper] slot {slot}: rejecting out-of-range frame {width}x{height} format={proxy_format}");
    }
    let n = if dims_ok { bytes } else { 0 };
    let motion_scale = neural_forge_protocol::motion::scales(hdr.mvec_scale_mode(), width, height);
    // Fixed addresses/capacity regardless of this frame's own width/height --
    // `FrameResources::new` decides for itself (per its own doc comment) whether
    // they're actually importable.
    let (proxy_region, answer_region) = shm.proxy_and_answer_regions(slot);

    // Nothing below the model's floor reaches NGX (or a prewarm): such a frame just echoes.
    let big_enough = width >= ngx::MIN_FEATURE_DIM && height >= ngx::MIN_FEATURE_DIM;
    let model_requested = dims_ok && big_enough && hdr.neural_enabled() && hdr.apply_model.load(Ordering::Relaxed) != 0;
    // Reserve the frame-sized Vulkan images as soon as the layer sees the game's real
    // swapchain, but do not enter the proprietary NGX runtime while NR is switched
    // off.  GTA is still bringing up its own GPU work at that point; calling
    // CreateFeature there has been observed to hang.  The resource reservation itself
    // is safe, makes later activation possible even after GTA fills VRAM, and
    // performs no model work or write-back.
    if dims_ok
        && big_enough
        && !model_requested
        && neural_forge_protocol::enums::proxy_format::is_8bit(proxy_format)
        && !frame_resources.as_ref().is_some_and(|f| f.matches(0, width, height, proxy_format))
    {
        if let Some(old) = frame_resources.take() {
            retire_frame_resources(old, device);
        }
        *frame_resources = frame::FrameResources::new(device, instance, physical_device, 0, width, height, proxy_format, proxy_region, answer_region);
        neural_forge_helper::log!("[helper] slot {slot}: prewarmed {}x{} frame resources: {}", width, height, frame_resources.is_some());
    }
    // One tuning per wanted pass; the chain builds a feature for each, so the header's pass count
    // (and per-pass overrides) decide how many times the model runs over the frame.
    let wanted_tunings: Vec<ngx::NgxTuning> = (0..hdr.resolved_passes() as usize).map(|i| hdr.resolve_pass(i).into()).collect();
    let live_passes = if model_requested {
        ngx::maintain_passes(
            snippet, device, queue, width, height, &wanted_tunings, hdr.rebuild_settle_ms.load(Ordering::Relaxed),
        )
    } else {
        0
    };
    let ready = live_passes > 0;
    if let Some(note) = snippet.take_failure_note() {
        hdr.set_helper_reason(&note);
    } else if ready {
        static CLEARED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        // Clears a stale failure reason once the model builds; a plain store per frame would churn
        // the seqlock for nothing.
        if !CLEARED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            hdr.set_helper_reason("");
        }
    }
    hdr.helper_features.store(live_passes as u32, Ordering::Relaxed);
    if let Some(ceiling) = snippet.pass_ceiling() {
        hdr.helper_pass_ceiling.store(ceiling as u32, Ordering::Relaxed);
    }
    if ready {
        hdr.model_up.store(1, Ordering::Relaxed);
    } else if model_requested && snippet.disabled {
        // `maintain_feature` only ever disables the snippet after a real, one-shot
        // `CreateFeature` attempt (see its own doc comment) -- worth surfacing in
        // status immediately rather than leaving `RUNNING` displayed forever after
        // the model is permanently unavailable.
        hdr.helper_state.store(neural_forge_protocol::enums::helper_state::MODEL_FAILED, Ordering::Relaxed);
    }
    // SAFETY: this helper exclusively owns this slot's request after observing its
    // own `seq_req`; slot 0's and slot 1's regions are disjoint fixed regions in the
    // mapping (`docs/PROTOCOL_V3_DESIGN.md`).
    let (proxy, answer) = unsafe { shm.frame_regions(slot, n) };

    // Real motion vectors, estimated on the GPU inside `evaluate` -- see `optical_flow.rs`.
    // Slot 0 only (protocol v3 never duplicated the motion payload for slot 1). The GUI's
    // "Estimate motion vectors" toggle is the only switch.
    let want_motion = slot == 0 && dims_ok && hdr.mvec_enabled() && neural_forge_protocol::enums::proxy_format::is_8bit(proxy_format);
    let scene_cut = motion.prepare(device, want_motion, proxy, width, height);

    let timing = if ready && neural_forge_protocol::enums::proxy_format::is_8bit(proxy_format) {
        (|| {
            if !frame_resources.as_ref().is_some_and(|f| f.matches(0, width, height, proxy_format)) {
                // `matches` returns `false` for a stalled instance too, so this branch
                // also catches "the previous instance's own fence wait timed out" --
                // `retire_frame_resources` leaks rather than destroys in exactly that
                // case, since a timeout no longer guarantees nothing is still
                // submitted the way an ordinary successful wait does.
                if let Some(old) = frame_resources.take() {
                    retire_frame_resources(old, device);
                }
                *frame_resources = frame::FrameResources::new(device, instance, physical_device, 0, width, height, proxy_format, proxy_region, answer_region);
            }
            let f = frame_resources.as_ref()?;
            let (Some(eval_fn), params) = (snippet.evaluate_feature_fn(), snippet.params()) else { return None };
            // Only a scene cut invalidates the model's history. Resetting whenever motion
            // was missing told the model "frame one" on every frame while motion was
            // unavailable, so its temporal history was never used.
            let reset_history = scene_cut;
            // Sharpness is per pass (the model reads it at evaluate); the header index of a pass
            // is its position in the chain, holes excluded only from the *built* handles, so the
            // pass numbers here are the first `live_passes` of the header's list.
            let chain: Vec<frame::ChainPass> = snippet
                .chain_handles()
                .into_iter()
                .enumerate()
                .map(|(i, handle)| frame::ChainPass {
                    handle,
                    reset: snippet.take_needs_reset(handle),
                    sharpness: hdr.resolve_pass(i).sharpness,
                })
                .collect();
            let flow = if want_motion { motion.session(instance, device, physical_device, flow_queue, width, height, hdr.mvec_quality.load(Ordering::Relaxed), scene_cut) } else { None };
            f.evaluate(device, queue, eval_fn, &chain, params, proxy, flow, motion_scale, reset_history, answer)
        })()
    } else {
        None
    };
    let evaluated = timing.is_some();
    if let Some(timing) = timing.as_ref() {
        motion.finished(device, timing);
    }
    if let Some(timing) = timing {
        store_ms(&hdr.helper_upload_ms_bits, timing.upload);
        store_ms(&hdr.helper_eval_ms_bits, timing.evaluate);
        store_ms(&hdr.helper_readback_ms_bits, timing.download);
    }
    if !evaluated {
        // Fail open: no real answer yet (feature still warming up, wrong proxy
        // format, or a guarded `EvaluateFeature` failure) -- echo the proxy straight
        // through so the transport round trip still completes with *something*
        // rather than stale/all-zero bytes.
        answer.copy_from_slice(proxy);
    }
    hdr.seq_ok.store(seq_req, Ordering::Relaxed);
    // The raster this answer is for, echoed before `seq_resp` so the layer can refuse an answer
    // for a different size (another swapchain's request, or one from before a resize) instead of
    // reading the wrong number of bytes. Slot 0 only: slot 1 has no such field.
    if slot == 0 {
        hdr.answered_w.store(if dims_ok { width } else { 0 }, Ordering::Relaxed);
        hdr.answered_h.store(if dims_ok { height } else { 0 }, Ordering::Relaxed);
    }
    if !helper_delay.is_zero() {
        std::thread::sleep(helper_delay);
    }
    hdr.seq_resp_slot(slot).store(seq_req, Ordering::Release);
    *frames += 1;
    neural_forge_protocol::store64(&hdr.helper_frames_lo, &hdr.helper_frames_hi, *frames);
    if neural_forge_helper::logging::sampled(*frames) || !evaluated {
        neural_forge_helper::log!(
            "[helper] slot {slot} frame {frames}: {width}x{height} evaluated={evaluated} passes={live_passes}/{} ceiling={:?}",
            wanted_tunings.len(), snippet.pass_ceiling()
        );
    }
}

/// A minimal Vulkan instance + device — just enough to hand NGX a live
/// `VkInstance`/`VkPhysicalDevice`/`VkDevice`, plus the one queue (family 0, index 0 --
/// every device has at least one family, and this is the same family `ngx`'s device
/// was created against) real per-frame work submits against.
/// Extensions a real, working reference implementation's own compiled helper
/// references (confirmed via `strings` on its binary, never its source -- run side
/// by side with ours this session, which is how the actual gap this closes was
/// found: our device previously enabled *none* of these). Requested only when
/// actually present in `vkEnumerateDeviceExtensionProperties` -- never assumed.
/// `VK_NVX_binary_import`/`VK_NVX_image_view_handle`/`VK_KHR_buffer_device_address`
/// in particular are the kind of NVIDIA-internal plumbing NGX's own shader/resource
/// management is plausibly built on; calling into it from a device that never
/// enabled them is the leading suspect for why `CreateFeature` behaved inconsistently
/// (a clean reject one run, an actual C++ exception the next) even after every
/// parameter this session could confirm from the same binary was added.
///
/// **2026-09-10 correction**: the reference binary `strings` was originally run
/// against was upstream's *native Linux* helper (per this project's own architecture,
/// upstream's own eventual roadmap target -- see the crate-level doc comment) --
/// `VK_EXT_external_memory_dma_buf`/`VK_KHR_external_memory_fd` are POSIX-specific
/// external-memory handle types that a real Linux Vulkan ICD legitimately exposes and
/// that reference binary legitimately used. `neural-forge-helper.exe` is not that: it is a
/// Windows binary running under Wine/Proton (this crate's current, documented, interim
/// architecture), and Wine's Vulkan implementation for Windows guest apps exposes the
/// Windows-shaped `VK_KHR_external_memory_win32` handle type, never the Linux `_fd`
/// ones -- requesting the Linux-specific pair here could never succeed no matter what
/// the real driver supports, a platform mismatch inherited from copying the reference
/// list without adjusting for which binary actually needed which handles. Found while
/// investigating a real, confirmed-white `EvaluateFeature` answer (see `CLAUDE.md`) --
/// this DLL performs real CUDA-Vulkan interop internally (`cuSurfObjectGetResourceDesc`
/// et al., confirmed via `strings` on the real DLL itself), which is exactly the kind
/// of external-memory-dependent operation a missing win32 handle type would degrade.
const WANTED_DEVICE_EXTENSIONS: &[&str] = &[
    "VK_EXT_debug_utils",
    "VK_EXT_external_memory_host",
    "VK_KHR_buffer_device_address",
    "VK_KHR_external_memory",
    "VK_KHR_external_memory_win32",
    "VK_KHR_push_descriptor",
    "VK_NV_optical_flow",
    "VK_NVX_binary_import",
    "VK_NVX_image_view_handle",
    // Added for real motion vectors (`optical_flow.rs`, 2026-09-17): both are
    // `VK_NV_optical_flow`'s own real dependencies (`vkCmdOpticalFlowExecuteNV`
    // synchronizes via `VK_KHR_synchronization2`'s timeline-barrier API;
    // `VK_KHR_format_feature_flags2` is the extended format-query struct optical
    // flow's own image-format negotiation uses). Requested only when actually
    // present, exactly like every other entry in this list -- their absence just
    // means `find_flow_family` below reports why there is no usable combination, not that
    // device creation itself is affected.
    "VK_KHR_synchronization2",
    "VK_KHR_format_feature_flags2",
];

/// The queue family/index that will actually run `vkCmdOpticalFlowExecuteNV`,
/// resolved once here (never re-queried per-frame) because Vulkan requires every
/// queue a device will ever use to be requested at `vkCreateDevice` time -- unlike
/// `frame::FrameResources`/NGX feature rebuilds, this can't be deferred to first use.
///
/// Returns `None` whenever optical flow genuinely isn't usable on this device: no
/// extension support, no driver-level feature support (a real, separate check from
/// "the extension string is present" -- `vkGetPhysicalDeviceFeatures2` is what
/// upstream's own `helper/main.cpp` checks too, not just extension enumeration), or
/// no queue family exposing `VK_QUEUE_OPTICAL_FLOW_BIT_NV`. Every caller treats that
/// exactly like a disabled feature -- this crate never fails to start NGX over it.
fn find_flow_family(instance: &ash::Instance, pd: vk::PhysicalDevice, enabled_extension_names: &[&str]) -> Result<u32, &'static str> {
    if !enabled_extension_names.contains(&"VK_NV_optical_flow") {
        return Err("VK_NV_optical_flow not exposed");
    }
    if !enabled_extension_names.contains(&"VK_KHR_synchronization2") {
        return Err("VK_KHR_synchronization2 not exposed");
    }
    let mut optical_features = vk::PhysicalDeviceOpticalFlowFeaturesNV::default();
    let mut sync_features = vk::PhysicalDeviceSynchronization2Features::default();
    // SAFETY: `pd` is a handle this process already enumerated; both feature structs
    // are default-initialized `VkBool32`-bearing structs, valid to chain and read.
    unsafe {
        instance.get_physical_device_features2(pd, &mut vk::PhysicalDeviceFeatures2::builder().push_next(&mut optical_features).push_next(&mut sync_features));
    }
    if optical_features.optical_flow == 0 {
        return Err("opticalFlow feature not supported");
    }
    if sync_features.synchronization2 == 0 {
        return Err("synchronization2 feature not supported");
    }
    // SAFETY: `pd` is a handle this process already enumerated.
    let families = unsafe { instance.get_physical_device_queue_family_properties(pd) };
    families.iter().position(|p| p.queue_flags.contains(vk::QueueFlags::OPTICAL_FLOW_NV | vk::QueueFlags::TRANSFER)).map(|i| i as u32)
        .ok_or("no optical-flow queue family")
}

/// `want_flow`: the motion-vector toggle as it stood at helper start. The optical-flow
/// queue and features are only requested then, so with the toggle off device creation
/// has exactly its pre-optical-flow shape. The last element says why there is or isn't
/// a flow queue, for the startup log.
fn create_vulkan_context(want_flow: bool) -> Option<(ash::Entry, ash::Instance, vk::PhysicalDevice, ash::Device, vk::Queue, Option<optical_flow::FlowQueue>, String)> {
    // SAFETY: dynamically loads `vulkan-1.dll` via the `loaded` feature; the usual
    // caveats of loading an arbitrary shared library apply and are accepted here the
    // same way every other `ash` consumer accepts them.
    let entry = unsafe { ash::Entry::load() }.ok()?;

    let app_info = vk::ApplicationInfo::builder().api_version(vk::API_VERSION_1_3);
    let create_info = vk::InstanceCreateInfo::builder().application_info(&app_info);
    // SAFETY: `create_info` is a valid, fully-populated `VkInstanceCreateInfo`.
    let instance = unsafe { entry.create_instance(&create_info, None) }.ok()?;

    // SAFETY: `instance` was just created above.
    let physical_devices = unsafe { instance.enumerate_physical_devices() }.ok()?;
    let physical_device = *physical_devices.iter().find(|&&pd| {
        // SAFETY: `pd` is one of the handles just enumerated.
        let props = unsafe { instance.get_physical_device_properties(pd) };
        props.vendor_id == 0x10DE // NVIDIA -- the model only ever runs on its own hardware.
    }).or(physical_devices.first())?;

    // SAFETY: `physical_device` is one of the handles just enumerated above.
    let available = unsafe { instance.enumerate_device_extension_properties(physical_device) }.unwrap_or_default();
    let available_names: std::collections::HashSet<String> = available
        .iter()
        .filter_map(|e| {
            // SAFETY: `extension_name` is a NUL-terminated C string the driver itself
            // populated; reading it as a CStr is exactly what every other `ash`
            // consumer does with this same field.
            unsafe { std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) }.to_str().ok().map(str::to_owned)
        })
        .collect();
    let enabled: Vec<&str> = WANTED_DEVICE_EXTENSIONS.iter().copied().filter(|e| available_names.contains(*e)).collect();
    neural_forge_helper::log!(
        "[helper] device extensions: {}/{} of the wanted set available: {:?}",
        enabled.len(),
        WANTED_DEVICE_EXTENSIONS.len(),
        enabled
    );
    let enabled_c: Vec<std::ffi::CString> = enabled.iter().map(|e| std::ffi::CString::new(*e).unwrap()).collect();
    let enabled_ptrs: Vec<*const std::ffi::c_char> = enabled_c.iter().map(|c| c.as_ptr()).collect();

    // `find_flow_family` re-checked here (not just trusted from a caller) is the one
    // source of truth for whether optical flow is genuinely usable -- extension
    // strings present *and* the driver-level features on, *and* a real queue family.
    // `None` means every path below behaves exactly as it did before this feature
    // existed: one queue on family 0, no extra features chained. Requesting a queue
    // family a second time in the same `DeviceQueueCreateInfo` array is invalid per
    // the Vulkan spec, so family 0 doubling as the flow family (common -- many
    // NVIDIA parts expose optical flow on their main graphics/compute family) is
    // handled by not adding a second entry for it, only chaining the extra features.
    //
    // Only when motion vectors are on: an unconditional second queue plus these features
    // were suspected (never proven) in a 2026-09-17 helper hang, so with the toggle off
    // device creation stays exactly as it was before optical flow existed.
    let (flow_family, flow_status) = if want_flow {
        match find_flow_family(&instance, physical_device, &enabled) {
            Ok(family) => (Some(family), format!("available (queue family {family})")),
            Err(why) => (None, format!("unavailable: {why}")),
        }
    } else {
        (None, "off (motion vectors disabled at helper start)".to_string())
    };
    let mut queue_infos = vec![vk::DeviceQueueCreateInfo::builder().queue_family_index(0).queue_priorities(&[1.0]).build()];
    if let Some(family) = flow_family {
        if family != 0 {
            queue_infos.push(vk::DeviceQueueCreateInfo::builder().queue_family_index(family).queue_priorities(&[1.0]).build());
        }
    }
    let mut optical_features = vk::PhysicalDeviceOpticalFlowFeaturesNV::builder().optical_flow(true);
    let mut sync_features = vk::PhysicalDeviceSynchronization2Features::builder().synchronization2(true);
    let mut device_create_info =
        vk::DeviceCreateInfo::builder().queue_create_infos(&queue_infos).enabled_extension_names(&enabled_ptrs);
    if flow_family.is_some() {
        device_create_info = device_create_info.push_next(&mut optical_features).push_next(&mut sync_features);
    }
    // SAFETY: `device_create_info` is valid; queue family 0 exists on every physical
    // device (the Vulkan spec guarantees at least one queue family), and `flow_family`
    // (when `Some`) was itself confirmed present by `find_flow_family`'s own
    // enumeration just above; `enabled_ptrs` point at only extensions just confirmed
    // present in `available_names`, and `enabled_c` (which owns the bytes they point
    // into), `optical_features`, `sync_features` all outlive this call.
    let device = unsafe { instance.create_device(physical_device, &device_create_info, None) }.ok()?;
    // SAFETY: `device` was just created with a queue on family 0, index 0 (always
    // requested above).
    let queue = unsafe { device.get_device_queue(0, 0) };
    let flow_queue = flow_family.map(|family| optical_flow::FlowQueue {
        family,
        // SAFETY: `family` is either 0 (already known valid) or was just requested
        // as this device's second queue above -- either way, index 0 on it is valid.
        queue: unsafe { device.get_device_queue(family, 0) },
    });

    Some((entry, instance, physical_device, device, queue, flow_queue, flow_status))
}
