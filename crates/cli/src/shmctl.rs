//! `neural-forge-cli shmctl` — raw status/set/toggle/capture against the live SHM header,
//! the real equivalent of upstream's own separate `neural-forge-shmctl` debug/introspection
//! tool (see the workspace `CLAUDE.md`'s "compared against a real, installed upstream
//! instance" entry: this project had no equivalent of it before now). Deliberately a
//! subcommand of `neural-forge-cli` rather than its own binary -- one fewer thing to build,
//! package, and document for what is fundamentally the same "attach to the mapping and
//! poke it" job `cmd_config`/the GUI's settings binding already do.
//!
//! `status`/`set`/`toggle` operate on the same 21-setting surface
//! `neural_forge_protocol::ShmHeader::persisted_settings`/`apply_persisted_setting` already
//! define (so a value changed here also gets written to `config.ini` on the GUI's next
//! save, the same as changing it from the GUI would), plus a handful of real,
//! genuinely useful fields that aren't user-facing "settings" in that sense --
//! `debug_view`/`capture_request` in particular, which is what makes this the actual
//! tool this project first used to visually confirm the composition pipeline produces
//! correct output (see `CLAUDE.md`'s "First confirmed *correct visual output*" entry).

use neural_forge_protocol::ShmHeader;
use std::sync::atomic::Ordering;

fn usage() {
    eprintln!(
        "usage: neural-forge-cli shmctl <status|set|toggle|capture|reset>\n\n\
         \x20 status              print every setting and live status field\n\
         \x20 set <name> <value>  set one setting (float fields take a decimal value)\n\
         \x20 toggle <name>       flip a 0/1-valued setting\n\
         \x20 capture [view]      dump the next frame's original+composited PNGs\n\
         \x20                     (see neural_forge_layer::dump); optional debug_view 0-3\n\
         \x20 reset               reset every setting to its default; preserves the\n\
         \x20                     live helper/layer session (see ShmHeader::reset_persisted_settings)\n\n\
         Respects $NEURAL_FORGE_SHM/$NEURAL_FORGE_UID, same as every other tool in this workspace."
    );
}

/// Extra fields worth real `set`/`toggle` access beyond what `persisted_settings`
/// covers. Just `capture_request` now -- `debug_view`/`apply_model`/`compare_mode`/
/// `hold_frame` moved into `persisted_settings` itself on 2026-09-10 (so a GUI change
/// to any of them now survives a reboot too), leaving this for the one field that
/// genuinely isn't a persisted preference: a one-shot trigger, not a setting.
fn extra_field<'a>(header: &'a ShmHeader, name: &str) -> Option<(&'a std::sync::atomic::AtomicU32, bool)> {
    Some(match name {
        "capture_request" => (&header.capture_request, false),
        _ => return None,
    })
}

fn helper_state_name(v: u32) -> &'static str {
    use neural_forge_protocol::enums::helper_state::*;
    match v {
        STARTING => "starting",
        NO_VULKAN => "no_vulkan",
        NO_BINARIES => "no_binaries",
        MODEL_FAILED => "model_failed",
        RUNNING => "running",
        STOPPED => "stopped",
        _ => "unknown",
    }
}

fn cmd_status(header: &ShmHeader) {
    println!("# live status");
    println!("helper_state={} ({})", header.helper_state.load(Ordering::Relaxed), helper_state_name(header.helper_state.load(Ordering::Relaxed)));
    println!("model_up={}", header.model_up.load(Ordering::Relaxed));
    let frames = (u64::from(header.helper_frames_hi.load(Ordering::Relaxed)) << 32) | u64::from(header.helper_frames_lo.load(Ordering::Relaxed));
    println!("helper_frames={frames}");
    println!("helper_upload_ms={}", f32::from_bits(header.helper_upload_ms_bits.load(Ordering::Relaxed)));
    println!("helper_eval_ms={}", f32::from_bits(header.helper_eval_ms_bits.load(Ordering::Relaxed)));
    println!("helper_readback_ms={}", f32::from_bits(header.helper_readback_ms_bits.load(Ordering::Relaxed)));
    let layer_frames = (u64::from(header.layer_frames_hi.load(Ordering::Relaxed)) << 32) | u64::from(header.layer_frames_lo.load(Ordering::Relaxed));
    println!("layer_frames={layer_frames}");
    println!("layer_ms={}", f32::from_bits(header.layer_ms_bits.load(Ordering::Relaxed)));
    println!("layer_composition_up={}", header.layer_composition_up.load(Ordering::Relaxed));
    println!("capture_request={}", header.capture_request.load(Ordering::Relaxed));
    println!("# settings (neural_forge_protocol::ShmHeader::persisted_settings)");
    for (name, is_float, bits) in header.persisted_settings() {
        if is_float {
            println!("{name}={}", f32::from_bits(bits));
        } else {
            println!("{name}={bits}");
        }
    }
}

/// Shared by `set`/`toggle`: resolves `name` against the 21 persisted settings first,
/// then the extra live-status fields, returning whether it's float-valued and its
/// current raw bits -- `None` if `name` isn't recognized by either.
fn resolve(header: &ShmHeader, name: &str) -> Option<(bool, u32)> {
    if let Some((_, is_float, bits)) = header.persisted_settings().into_iter().find(|(n, ..)| *n == name) {
        return Some((is_float, bits));
    }
    extra_field(header, name).map(|(field, is_float)| (is_float, field.load(Ordering::Relaxed)))
}

fn store(header: &ShmHeader, name: &str, bits: u32) -> bool {
    if header.persisted_settings().iter().any(|(n, ..)| *n == name) {
        header.apply_persisted_setting(name, bits);
        return true;
    }
    if let Some((field, _)) = extra_field(header, name) {
        field.store(bits, Ordering::Relaxed);
        return true;
    }
    false
}

fn cmd_set(header: &ShmHeader, name: &str, value: &str) -> bool {
    let Some((is_float, _)) = resolve(header, name) else {
        eprintln!("shmctl set: unknown setting {name:?}");
        return false;
    };
    let bits = if is_float {
        match value.parse::<f32>() {
            Ok(v) => v.to_bits(),
            Err(_) => {
                eprintln!("shmctl set: {name} takes a decimal value, got {value:?}");
                return false;
            }
        }
    } else {
        match value.parse::<u32>() {
            Ok(v) => v,
            Err(_) => {
                eprintln!("shmctl set: {name} takes an integer value, got {value:?}");
                return false;
            }
        }
    };
    store(header, name, bits);
    println!("{name}={value}");
    true
}

fn cmd_toggle(header: &ShmHeader, name: &str) -> bool {
    let Some((is_float, bits)) = resolve(header, name) else {
        eprintln!("shmctl toggle: unknown setting {name:?}");
        return false;
    };
    if is_float {
        eprintln!("shmctl toggle: {name} is a float-valued setting, use `set` instead");
        return false;
    }
    let new = u32::from(bits == 0);
    store(header, name, new);
    println!("{name}={new}");
    true
}

fn cmd_capture(header: &ShmHeader, view: Option<&str>) -> bool {
    if let Some(view) = view {
        let Ok(mode) = view.parse::<u32>() else {
            eprintln!("shmctl capture: debug_view must be 0-3, got {view:?}");
            return false;
        };
        header.debug_view.store(mode, Ordering::Relaxed);
    }
    header.capture_request.store(1, Ordering::Relaxed);
    println!("capture_request set -- check $XDG_DATA_HOME/neural-forge/captures on the layer's next present");
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_finds_a_persisted_setting() {
        let header = ShmHeader::default();
        header.init_defaults();
        let (is_float, bits) = resolve(&header, "colour_strength").expect("colour_strength should resolve");
        assert!(is_float);
        assert_eq!(f32::from_bits(bits), 1.0);
    }

    #[test]
    fn resolve_finds_an_extra_field() {
        let header = ShmHeader::default();
        let (is_float, bits) = resolve(&header, "capture_request").expect("capture_request should resolve");
        assert!(!is_float);
        assert_eq!(bits, 0);
    }

    #[test]
    fn resolve_finds_debug_view_via_persisted_settings() {
        // debug_view/apply_model/compare_mode/hold_frame moved into
        // ShmHeader::persisted_settings on 2026-09-10 -- confirms this module's
        // `extra_field` no longer needs (and no longer has) a special case for them.
        let header = ShmHeader::default();
        let (is_float, _) = resolve(&header, "debug_view").expect("debug_view should resolve");
        assert!(!is_float);
    }

    #[test]
    fn resolve_rejects_an_unknown_name() {
        let header = ShmHeader::default();
        assert!(resolve(&header, "not_a_real_setting").is_none());
    }

    #[test]
    fn store_writes_through_persisted_settings() {
        let header = ShmHeader::default();
        assert!(store(&header, "colour_strength", 0.25f32.to_bits()));
        assert_eq!(header.colour_strength_bits.load(Ordering::Relaxed), 0.25f32.to_bits());
    }

    #[test]
    fn store_writes_through_extra_fields() {
        let header = ShmHeader::default();
        assert!(store(&header, "capture_request", 1));
        assert_eq!(header.capture_request.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn store_rejects_an_unknown_name() {
        let header = ShmHeader::default();
        assert!(!store(&header, "not_a_real_setting", 1));
    }

    #[test]
    fn cmd_toggle_flips_a_boolean_field_both_ways() {
        let header = ShmHeader::default();
        assert!(cmd_toggle(&header, "apply_model"));
        assert_eq!(header.apply_model.load(Ordering::Relaxed), 1);
        assert!(cmd_toggle(&header, "apply_model"));
        assert_eq!(header.apply_model.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn cmd_toggle_refuses_a_float_field() {
        let header = ShmHeader::default();
        assert!(!cmd_toggle(&header, "colour_strength"));
    }

    #[test]
    fn cmd_set_rejects_a_non_numeric_value_for_a_float_field() {
        let header = ShmHeader::default();
        assert!(!cmd_set(&header, "colour_strength", "not-a-number"));
    }

    #[test]
    fn helper_state_name_covers_every_real_state() {
        use neural_forge_protocol::enums::helper_state::*;
        for state in [STARTING, NO_VULKAN, NO_BINARIES, MODEL_FAILED, RUNNING, STOPPED] {
            assert_ne!(helper_state_name(state), "unknown");
        }
        assert_eq!(helper_state_name(9999), "unknown");
    }
}

pub fn run(args: &[String]) -> std::process::ExitCode {
    let Some(mapping) = neural_forge_protocol::mapping::open() else {
        eprintln!("shmctl: failed to open the SHM mapping (see $NEURAL_FORGE_SHM/$NEURAL_FORGE_UID)");
        return std::process::ExitCode::FAILURE;
    };
    let header = mapping.header();

    let ok = match args.first().map(String::as_str) {
        Some("status") => {
            cmd_status(header);
            true
        }
        Some("set") => match (args.get(1), args.get(2)) {
            (Some(name), Some(value)) => cmd_set(header, name, value),
            _ => {
                eprintln!("usage: neural-forge-cli shmctl set <name> <value>");
                false
            }
        },
        Some("toggle") => match args.get(1) {
            Some(name) => cmd_toggle(header, name),
            None => {
                eprintln!("usage: neural-forge-cli shmctl toggle <name>");
                false
            }
        },
        Some("capture") => cmd_capture(header, args.get(1).map(String::as_str)),
        Some("reset") => {
            header.reset_persisted_settings();
            println!("settings reset to defaults; helper/layer session preserved");
            true
        }
        _ => {
            usage();
            false
        }
    };
    if ok {
        std::process::ExitCode::SUCCESS
    } else {
        std::process::ExitCode::FAILURE
    }
}
