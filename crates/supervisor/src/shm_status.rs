//! The live channel's status as text: every live status field and every persisted setting, one
//! `name=value` line each. `neural-forge-cli shmctl status` prints it and the diagnostic report
//! (`crate::report`) stores it, so both show the same thing.

use std::fmt::Write;
use std::sync::atomic::Ordering;

use neural_forge_protocol::ShmHeader;

pub fn server_state_name(v: u32) -> &'static str {
    use neural_forge_protocol::enums::server_state::*;
    match v {
        MODEL_FAILED => "model_failed",
        RUNNING => "running",
        STOPPED => "stopped",
        _ => "unknown",
    }
}

/// What `preupscale_state` (`neural_forge_protocol::enums::preupscale_state`) means, as `shmctl status` and the
/// report say it.
pub fn preupscale_state_name(v: u32) -> &'static str {
    use neural_forge_protocol::enums::preupscale_state::*;
    match v {
        OFF => "off",
        WAITING => "waiting for DLSS input",
        HOLDING => "holding before the upscaler",
        HOLDING_AFTER => "holding after DLSS",
        HOLDING_BOTH => "holding before the upscaler and after DLSS",
        IDENTIFIED => "DLSS input identified, not holding",
        PAUSED => "paused with DLSS (a menu or loading screen); the model resumes with it",
        NO_WRAPPER => "off: this game uses DLSS but isn't on the Neural Forge Proton (switch it on in Neural Forge to set it)",
        _ => "unknown",
    }
}

/// The channel's status text and the game it names, looked at read-only
/// (`neural_forge_protocol::mapping::open_existing`): `shmctl status`, the report and the GUI's
/// "Save report" never create the channel, apply settings to it or write to it. Without a
/// channel, or with one that cannot be read, the text says so and the game is empty.
pub fn look(cfg: &crate::Config) -> (String, String) {
    let path = crate::channel_path(cfg);
    match neural_forge_protocol::mapping::open_existing(&path) {
        Ok(Some(mapping)) => (status_text_with(mapping.header(), crate::desktop_hdr::read().as_deref()), mapping.header().game_name()),
        Ok(None) => (format!("no channel at {path}: no game and no Neural Forge window has opened it since the computer started\n"), String::new()),
        Err(e) => (format!("channel not readable: {e}: {path}\n"), String::new()),
    }
}

/// `header`'s live status fields, then its settings (`ShmHeader::persisted_settings`).
pub fn status_text(header: &ShmHeader) -> String {
    status_text_with(header, None)
}

/// The HDR verdict as one line: the layer's (`ShmHeader::hdr_verdict`) and the desktop's output state
/// (`crate::desktop_hdr::read`), when known.
pub fn hdr_line(header: &ShmHeader, desktop: Option<&str>) -> String {
    let game = header.hdr_verdict();
    let game = if game.is_empty() { "no game swapchain seen yet".to_string() } else { game };
    match desktop {
        Some(d) => format!("{game}; desktop: {d}"),
        None => game,
    }
}

/// [`status_text`] with the desktop's HDR state (`crate::desktop_hdr::read`) in the `hdr_verdict` line.
pub fn status_text_with(header: &ShmHeader, desktop: Option<&str>) -> String {
    let mut s = String::new();
    // Writing to a `String` cannot fail.
    let mut line = |args: std::fmt::Arguments| {
        let _ = s.write_fmt(args);
        s.push('\n');
    };
    line(format_args!("# live status"));
    let server_state = header.server_state.load(Ordering::Relaxed);
    line(format_args!("server_state={server_state} ({})", server_state_name(server_state)));
    line(format_args!("model_up={}", header.model_up.load(Ordering::Relaxed)));
    let frames = (u64::from(header.server_frames_hi.load(Ordering::Relaxed)) << 32) | u64::from(header.server_frames_lo.load(Ordering::Relaxed));
    line(format_args!("server_frames={frames}"));
    line(format_args!("server_upload_ms={}", f32::from_bits(header.server_upload_ms_bits.load(Ordering::Relaxed))));
    line(format_args!("server_eval_ms={}", f32::from_bits(header.server_eval_ms_bits.load(Ordering::Relaxed))));
    line(format_args!("server_readback_ms={}", f32::from_bits(header.server_readback_ms_bits.load(Ordering::Relaxed))));
    line(format_args!("server_busy_ms={}", f64::from(header.server_busy_us.load(Ordering::Relaxed)) / 1000.0));
    let layer_frames = (u64::from(header.layer_frames_hi.load(Ordering::Relaxed)) << 32) | u64::from(header.layer_frames_lo.load(Ordering::Relaxed));
    line(format_args!("layer_frames={layer_frames}"));
    line(format_args!("layer_ms={}", f32::from_bits(header.layer_ms_bits.load(Ordering::Relaxed))));
    line(format_args!("layer_capture_gpu_ms={}", f32::from_bits(header.layer_capture_gpu_ms_bits.load(Ordering::Relaxed))));
    line(format_args!("layer_compose_gpu_ms={}", f32::from_bits(header.layer_compose_gpu_ms_bits.load(Ordering::Relaxed))));
    let preupscale = header.preupscale_state.load(Ordering::Relaxed);
    line(format_args!("preupscale_state={preupscale} ({})", preupscale_state_name(preupscale)));
    line(format_args!("preupscale_reason={}", header.preupscale_reason()));
    line(format_args!("layer_reason={}", header.layer_reason()));
    line(format_args!("native_running={}", header.native_running.load(Ordering::Relaxed)));
    let hdr = header.hdr_detected.load(Ordering::Relaxed);
    let hdr_active = header.hdr_active.load(Ordering::Relaxed);
    line(format_args!("swapchain_hdr={hdr} ({})", {
        use neural_forge_protocol::enums::hdr_kind::*;
        match hdr {
            NONE => "SDR, or no game yet",
            LINEAR_FP16 => "HDR, linear FP16: presented untouched",
            PQ10 if hdr_active == PQ10 => "HDR10, PQ: the network runs on an SDR-mapped copy",
            PQ10 => "HDR10, PQ: presented untouched",
            _ => "unknown",
        }
    }));
    line(format_args!("hdr_verdict={}", hdr_line(header, desktop)));
    line(format_args!("second_gpu_present={}", header.second_gpu_present.load(Ordering::Relaxed)));
    line(format_args!("second_gpu_present_line={}", header.second_gpu_present_line()));
    line(format_args!("device_lost_at={}", header.device_lost_at.load(Ordering::Relaxed)));
    line(format_args!("preupscale_extent={}x{}", header.preupscale_width.load(Ordering::Relaxed), header.preupscale_height.load(Ordering::Relaxed)));
    line(format_args!("preupscale_hold_ms={}", f32::from_bits(header.preupscale_hold_ms_bits.load(Ordering::Relaxed))));
    line(format_args!("preupscale_misses={}", header.preupscale_misses.load(Ordering::Relaxed)));
    line(format_args!("ngx_records={}", header.ngx_records.load(Ordering::Relaxed)));
    line(format_args!("ngx_matched={}", header.ngx_matched.load(Ordering::Relaxed)));
    line(format_args!("preupscale_network_gpu_ms={}", f32::from_bits(header.preupscale_network_gpu_ms_bits.load(Ordering::Relaxed))));
    line(format_args!("layer_measured_white={}", f32::from_bits(header.layer_measured_white_bits.load(Ordering::Relaxed))));
    line(format_args!("layer_composition_up={}", header.layer_composition_up.load(Ordering::Relaxed)));
    match neural_forge_protocol::capture::pending(header) {
        Some(r) => line(format_args!("capture_request=id {} for {} frames, waiting for the game", r.id, r.frames)),
        None => line(format_args!("capture_request=none")),
    }
    if let Some(o) = neural_forge_protocol::capture::result(header) {
        line(format_args!("capture_result=id {}: {}", o.id, o.message()));
        for path in o.paths() {
            line(format_args!("capture_path={path}"));
        }
    }
    line(format_args!("preupscale_dump_request={}", header.preupscale_dump_request.load(Ordering::Relaxed)));
    line(format_args!(
        "edit_cache: refresh_holds={} cached_holds={} rejected={:.2}% gpu_ms={:.3}",
        header.cache_refresh_holds.load(Ordering::Relaxed),
        header.cache_cached_holds.load(Ordering::Relaxed),
        f64::from(header.cache_rejected.load(Ordering::Relaxed)) / 100.0,
        f32::from_bits(header.cache_gpu_ms_bits.load(Ordering::Relaxed))
    ));
    line(format_args!("scene_cuts={}", header.scene_cuts.load(Ordering::Relaxed)));
    line(format_args!("# settings (neural_forge_protocol::ShmHeader::persisted_settings)"));
    for (name, is_float, bits) in header.persisted_settings() {
        if is_float {
            line(format_args!("{name}={}", f32::from_bits(bits)));
        } else {
            line(format_args!("{name}={bits}"));
        }
    }
    // The one setting that is text.
    line(format_args!("{}={}", neural_forge_protocol::persist::HUD_REGIONS, neural_forge_protocol::HudRegion::list_text(&header.hud_settings().regions)));
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn server_state_name_covers_every_real_state() {
        use neural_forge_protocol::enums::server_state::*;
        for state in [MODEL_FAILED, RUNNING, STOPPED] {
            assert_ne!(server_state_name(state), "unknown");
        }
        assert_eq!(server_state_name(9999), "unknown");
    }

    #[test]
    fn preupscale_state_name_covers_every_real_state() {
        use neural_forge_protocol::enums::preupscale_state::*;
        let names: Vec<_> = [OFF, WAITING, HOLDING, HOLDING_AFTER, HOLDING_BOTH, IDENTIFIED, PAUSED, NO_WRAPPER].map(preupscale_state_name).into();
        assert!(!names.contains(&"unknown"), "{names:?}");
        assert_eq!(names.iter().collect::<std::collections::HashSet<_>>().len(), names.len(), "each its own: {names:?}");
        assert_eq!(preupscale_state_name(IDENTIFIED), "DLSS input identified, not holding");
        assert_eq!(preupscale_state_name(NO_WRAPPER + 1), "unknown");
    }

    /// DLSS identified with every frame refused (#51): the state says so, and the next line says why.
    #[test]
    fn the_status_says_why_dlss_identified_is_not_held() {
        let header = ShmHeader::default();
        header.init_defaults();
        let text = status_text(&header);
        assert!(text.lines().any(|l| l == "preupscale_reason="), "empty while nothing is refused: {text}");
        header.preupscale_state.store(neural_forge_protocol::enums::preupscale_state::IDENTIFIED, Ordering::Relaxed);
        header.set_preupscale_reason("no hold inside DLSS's buffer: the colour input is 1920x1080, the identified one 1280x720");
        let text = status_text(&header);
        let lines: Vec<&str> = text.lines().collect();
        let at = lines.iter().position(|l| *l == "preupscale_state=5 (DLSS input identified, not holding)").expect(&text);
        assert_eq!(lines[at + 1], "preupscale_reason=no hold inside DLSS's buffer: the colour input is 1920x1080, the identified one 1280x720");
    }

    #[test]
    fn the_status_lists_live_fields_then_every_setting() {
        let header = ShmHeader::default();
        header.init_defaults();
        let id = neural_forge_protocol::capture::request(&header, 1).unwrap();
        neural_forge_protocol::capture::publish(
            &header,
            &neural_forge_protocol::capture::Outcome {
                id: 9,
                state: neural_forge_protocol::capture::state::WRITTEN,
                done: 1,
                total: 1,
                composed: 1,
                paths: "/c/1-original.png\n/c/1-composited.png".into(),
                ..Default::default()
            },
        );
        header.preupscale_dump_request.store(16, Ordering::Relaxed);
        header.cache_refresh_holds.store(30, Ordering::Relaxed);
        header.cache_cached_holds.store(29, Ordering::Relaxed);
        header.cache_rejected.store(1234, Ordering::Relaxed);
        header.cache_gpu_ms_bits.store(0.25f32.to_bits(), Ordering::Relaxed);
        header.scene_cuts.store(3, Ordering::Relaxed);
        header.hdr_detected.store(neural_forge_protocol::enums::hdr_kind::PQ10, Ordering::Relaxed);
        let text = status_text(&header);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], "# live status");
        assert!(lines.contains(&format!("capture_request=id {id} for 1 frames, waiting for the game").as_str()), "{text}");
        assert!(lines.contains(&"capture_result=id 9: Captured"), "{text}");
        assert!(lines.contains(&"capture_path=/c/1-composited.png"), "{text}");
        assert!(lines.contains(&"preupscale_dump_request=16"), "{text}");
        assert!(lines.contains(&"edit_cache: refresh_holds=30 cached_holds=29 rejected=12.34% gpu_ms=0.250"), "{text}");
        assert!(lines.contains(&"scene_cuts=3"), "{text}");
        assert!(lines.contains(&"swapchain_hdr=2 (HDR10, PQ: presented untouched)"), "{text}");
        assert!(lines.contains(&"hdr_verdict=no game swapchain seen yet"), "{text}");
        header.set_hdr_verdict("game outputs HDR10 (swapchain A2B10G10R10_UNORM_PACK32 HDR10_ST2084_EXT)");
        let with = status_text_with(&header, Some("GNOME: HDMI-1 HDR on (BT.2100)"));
        assert!(
            with.lines()
                .any(|l| l == "hdr_verdict=game outputs HDR10 (swapchain A2B10G10R10_UNORM_PACK32 HDR10_ST2084_EXT); desktop: GNOME: HDMI-1 HDR on (BT.2100)"),
            "{with}"
        );
        header.hdr_active.store(neural_forge_protocol::enums::hdr_kind::PQ10, Ordering::Relaxed);
        assert!(status_text(&header).lines().any(|l| l == "swapchain_hdr=2 (HDR10, PQ: the network runs on an SDR-mapped copy)"));
        assert!(lines.contains(&"intensity=1"), "{text}");
        assert!(lines.contains(&"skin_structure=-1"), "{text}");
        let settings_at = lines.iter().position(|l| l.starts_with("# settings")).expect("a settings section");
        assert_eq!(lines.len() - settings_at - 1, header.persisted_settings().len() + 1, "and hud_regions");
        assert!(lines.contains(&"hud_regions="), "{text}");
        assert!(text.ends_with('\n'));
    }
}
