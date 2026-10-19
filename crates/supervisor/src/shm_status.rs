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

/// The channel's status text and the game it names, looked at read-only
/// (`neural_forge_protocol::mapping::open_existing`): `shmctl status`, the report and the GUI's
/// "Save report" never create the channel, apply settings to it or write to it. Without a
/// channel, or with one that cannot be read, the text says so and the game is empty.
pub fn look(cfg: &crate::Config) -> (String, String) {
    let path = crate::channel_path(cfg);
    match neural_forge_protocol::mapping::open_existing(&path) {
        Ok(Some(mapping)) => (status_text(mapping.header()), mapping.header().game_name()),
        Ok(None) => (format!("no channel at {path}: no game and no Neural Forge window has opened it since the computer started\n"), String::new()),
        Err(e) => (format!("channel not readable: {e}: {path}\n"), String::new()),
    }
}

/// `header`'s live status fields, then its settings (`ShmHeader::persisted_settings`).
pub fn status_text(header: &ShmHeader) -> String {
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
    line(format_args!("preupscale_state={preupscale} ({})", {
        use neural_forge_protocol::enums::preupscale_state::*;
        match preupscale {
            OFF => "off",
            WAITING => "waiting for DLSS input",
            HOLDING => "holding",
            _ => "unknown",
        }
    }));
    line(format_args!("layer_reason={}", header.layer_reason()));
    line(format_args!("native_running={}", header.native_running.load(Ordering::Relaxed)));
    line(format_args!("device_lost_at={}", header.device_lost_at.load(Ordering::Relaxed)));
    line(format_args!("preupscale_extent={}x{}", header.preupscale_width.load(Ordering::Relaxed), header.preupscale_height.load(Ordering::Relaxed)));
    line(format_args!("preupscale_hold_ms={}", f32::from_bits(header.preupscale_hold_ms_bits.load(Ordering::Relaxed))));
    line(format_args!("preupscale_misses={}", header.preupscale_misses.load(Ordering::Relaxed)));
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
    fn the_status_lists_live_fields_then_every_setting() {
        let header = ShmHeader::default();
        header.init_defaults();
        let id = neural_forge_protocol::capture::request(&header, 1).unwrap();
        neural_forge_protocol::capture::publish(&header, &neural_forge_protocol::capture::Outcome {
            id: 9,
            state: neural_forge_protocol::capture::state::WRITTEN,
            done: 1,
            total: 1,
            composed: 1,
            paths: "/c/1-original.png\n/c/1-composited.png".into(),
            ..Default::default()
        });
        header.preupscale_dump_request.store(16, Ordering::Relaxed);
        header.cache_refresh_holds.store(30, Ordering::Relaxed);
        header.cache_cached_holds.store(29, Ordering::Relaxed);
        header.cache_rejected.store(1234, Ordering::Relaxed);
        header.cache_gpu_ms_bits.store(0.25f32.to_bits(), Ordering::Relaxed);
        header.scene_cuts.store(3, Ordering::Relaxed);
        let text = status_text(&header);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], "# live status");
        assert!(lines.contains(&format!("capture_request=id {id} for 1 frames, waiting for the game").as_str()), "{text}");
        assert!(lines.contains(&"capture_result=id 9: Captured"), "{text}");
        assert!(lines.contains(&"capture_path=/c/1-composited.png"), "{text}");
        assert!(lines.contains(&"preupscale_dump_request=16"), "{text}");
        assert!(lines.contains(&"edit_cache: refresh_holds=30 cached_holds=29 rejected=12.34% gpu_ms=0.250"), "{text}");
        assert!(lines.contains(&"scene_cuts=3"), "{text}");
        assert!(lines.contains(&"intensity=1"), "{text}");
        assert!(lines.contains(&"skin_structure=-1"), "{text}");
        let settings_at = lines.iter().position(|l| l.starts_with("# settings")).expect("a settings section");
        assert_eq!(lines.len() - settings_at - 1, header.persisted_settings().len());
        assert!(text.ends_with('\n'));
    }
}
