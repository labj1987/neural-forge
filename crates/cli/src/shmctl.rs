//! `neural-forge-cli shmctl`: raw status/set/toggle/capture against the live shared-memory header,
//! for debugging and scripted measurements. A subcommand rather than its own binary: it is the same
//! "attach to the mapping and poke it" job the GUI's settings binding does.
//!
//! `status`/`set`/`toggle` operate on the setting surface
//! `neural_forge_protocol::ShmHeader::persisted_settings`/`apply_persisted_setting` define (a value
//! changed here is saved to `config.ini`, as a change from the GUI is). `capture` asks the game's
//! layer for a capture (`neural_forge_protocol::capture`) and reports what became of it, which
//! makes this the tool for confirming the composition visually.

use neural_forge_protocol::ShmHeader;
use std::sync::atomic::Ordering;

pub fn usage() {
    eprintln!(
        "usage: neural-forge-cli shmctl <status|set|toggle|capture|dump-cache|reset>\n\n\
         \x20 status              print every setting and live status field\n\
         \x20 set <name> <value>  set one setting (float fields take a decimal value)\n\
         \x20 toggle <name>       flip a 0/1-valued setting\n\
         \x20 capture [--frames N] [--view V]\n\
         \x20                     save the next presented frame (or the next N, at most\n\
         \x20                     {max_frames}, as a series-<ms>/ folder) as original+composited\n\
         \x20                     PNGs under $XDG_DATA_HOME/neural-forge/captures/, and wait\n\
         \x20                     for the game's layer to say what it wrote; --view sets\n\
         \x20                     debug_view ({view_min}-{view_max}) first\n\
         \x20 capture --cancel    stop the capture the channel is waiting on or running\n\
         \x20 dump-cache [--holds K]  dump the next K model-mode holds before the upscaler\n\
         \x20                     (default 16, at most 64) to $XDG_DATA_HOME/neural-forge/\n\
         \x20                     captures/preupscale-cache-<ms>/ (docs/ARCHITECTURE.md)\n\
         \x20 reset               reset every setting to its default; preserves the\n\
         \x20                     live layer session (see ShmHeader::reset_persisted_settings)\n\n\
         Opens config.ini's shm= channel, else $NEURAL_FORGE_SHM, else the default under\n\
         /tmp/neural-forge-$UID -- the same one the layer uses.",
        max_frames = neural_forge_protocol::capture::MAX_FRAMES,
        view_min = debug_view_bounds().0,
        view_max = debug_view_bounds().1,
    );
}

fn cmd_status(header: &ShmHeader) {
    print!("{}", neural_forge_supervisor::shm_status::status_text(header));
}

/// Shared by `set`/`toggle`: resolves `name` against the persisted settings, returning whether it's
/// float-valued and its current raw bits -- `None` if `name` isn't one.
fn resolve(header: &ShmHeader, name: &str) -> Option<(bool, u32)> {
    header.persisted_settings().into_iter().find(|(n, ..)| *n == name).map(|(_, is_float, bits)| (is_float, bits))
}

/// Writes one setting and bumps the sequence numbers the layer watches. The
/// `tuning_seq` bump also marks the header as configured: left at 0 (a header nothing had
/// applied `config.ini` to yet), the next `start()` would overwrite this value from it.
fn store(header: &ShmHeader, name: &str, bits: u32) -> bool {
    if header.persisted_settings().iter().any(|(n, ..)| *n == name) {
        if !header.apply_persisted_setting(name, bits) {
            return false;
        }
        header.tuning_seq.fetch_add(1, Ordering::Relaxed);
        header.control_seq.fetch_add(1, Ordering::Relaxed);
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
    if let Some((min, max)) = neural_forge_protocol::setting_bounds(name) {
        let v = if is_float { f32::from_bits(bits) } else { bits as f32 };
        if !(min..=max).contains(&v) {
            eprintln!("shmctl set: {name} takes a value from {min} to {max}, got {value:?}");
            return false;
        }
    }
    if !store(header, name, bits) {
        eprintln!("shmctl set: {name} was not stored");
        return false;
    }
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

/// `debug_view`'s range, from `SETTING_BOUNDS`.
fn debug_view_bounds() -> (u32, u32) {
    neural_forge_protocol::setting_bounds("debug_view").map_or((0, 0), |(min, max)| (min as u32, max as u32))
}

/// What `capture` was asked to do.
#[derive(Debug, PartialEq, Eq)]
enum CaptureArgs {
    Take { frames: u32, view: Option<u32> },
    Cancel,
}

fn capture_args(args: &[String]) -> Result<CaptureArgs, String> {
    let (mut frames, mut view, mut cancel) = (None, None, false);
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--frames" if frames.is_none() => {
                let max = neural_forge_protocol::capture::MAX_FRAMES;
                let n = it.next().and_then(|v| v.parse::<u32>().ok()).filter(|n| (1..=max).contains(n));
                frames = Some(n.ok_or_else(|| format!("shmctl capture: --frames takes a count from 1 to {max}"))?);
            }
            "--view" if view.is_none() => {
                let (min, max) = debug_view_bounds();
                let v = it.next().and_then(|v| v.parse::<u32>().ok()).filter(|v| (min..=max).contains(v));
                view = Some(v.ok_or_else(|| format!("shmctl capture: --view takes a debug view from {min} to {max}"))?);
            }
            "--cancel" => cancel = true,
            other => return Err(format!("shmctl capture: unexpected argument {other:?}")),
        }
    }
    match (cancel, frames, view) {
        (true, None, None) => Ok(CaptureArgs::Cancel),
        (true, ..) => Err("shmctl capture: --cancel takes no other option".into()),
        (false, frames, view) => Ok(CaptureArgs::Take { frames: frames.unwrap_or(1), view }),
    }
}

/// How long a request may wait for a game to take it (then it is withdrawn), and how long a taken
/// capture may go without finishing: a base, plus a second a frame for a series.
const TAKE_WAIT: std::time::Duration = std::time::Duration::from_secs(5);
fn result_wait(frames: u32) -> std::time::Duration {
    std::time::Duration::from_secs(30 + u64::from(frames))
}

/// Waits for capture `id`'s outcome: withdraws the request when no game took it within `take_wait`,
/// gives up after `result_wait`. Prints a series' progress as it goes.
fn wait_for_capture(header: &ShmHeader, id: u32, take_wait: std::time::Duration, result_wait: std::time::Duration) -> Result<neural_forge_protocol::capture::Outcome, String> {
    use neural_forge_protocol::capture;
    let start = std::time::Instant::now();
    let mut said = 0;
    loop {
        if let Some(outcome) = capture::result(header).filter(|o| o.id == id) {
            if outcome.finished() {
                return Ok(outcome);
            }
            if outcome.total > 1 && outcome.done >= said + 10 {
                said = outcome.done;
                eprintln!("shmctl capture: {} of {} frames written", outcome.done, outcome.total);
            }
        }
        let taken = capture::pending(header).is_none_or(|r| r.id != id);
        if !taken && start.elapsed() >= take_wait && capture::cancel(header, id) == capture::Cancelled::Withdrawn {
            return Err("no game took the capture (withdrawn): start a game with Neural Forge on and wait until it is past its loading screen".into());
        }
        if start.elapsed() >= result_wait {
            return Err("the game took the capture but did not finish it in time (its log may say why); `shmctl capture --cancel` stops it".into());
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
}

fn cmd_capture(header: &ShmHeader, args: &[String]) -> bool {
    use neural_forge_protocol::capture;
    let parsed = match capture_args(args) {
        Ok(parsed) => parsed,
        Err(e) => {
            eprintln!("{e}");
            return false;
        }
    };
    let (frames, view) = match parsed {
        CaptureArgs::Cancel => {
            let waiting = capture::pending(header).map(|r| r.id);
            let running = capture::result(header).filter(|o| o.state == capture::state::RUNNING).map(|o| o.id);
            let Some(id) = waiting.or(running) else {
                println!("shmctl capture: no capture is waiting or running");
                return true;
            };
            match capture::cancel(header, id) {
                capture::Cancelled::Withdrawn => println!("capture {id} withdrawn before a game took it"),
                capture::Cancelled::Signalled => println!("capture {id} stops at the game's next present"),
            }
            return true;
        }
        CaptureArgs::Take { frames, view } => (frames, view),
    };
    if let Some(view) = view {
        if !store(header, "debug_view", view) {
            eprintln!("shmctl capture: debug_view was not stored");
            return false;
        }
    }
    let id = match capture::request(header, frames) {
        Ok(id) => id,
        Err(e) => {
            eprintln!("shmctl capture: {e} (`shmctl capture --cancel` withdraws it)");
            return false;
        }
    };
    match wait_for_capture(header, id, TAKE_WAIT, result_wait(frames)) {
        Ok(outcome) => {
            println!("{}", outcome.message());
            for path in outcome.paths() {
                println!("  {path}");
            }
            outcome.state == capture::state::WRITTEN
        }
        Err(e) => {
            eprintln!("shmctl capture: {e}");
            false
        }
    }
}

/// `dump-cache [--holds K]`: the hold count it asks for, or why the arguments are wrong.
fn dump_cache_holds(args: &[String]) -> Result<u32, String> {
    use neural_forge_protocol::cache_dump::{parse_holds, DEFAULT_HOLDS, MAX_HOLDS};
    match args {
        [] => Ok(DEFAULT_HOLDS),
        [flag, k] if flag == "--holds" => parse_holds(k).ok_or_else(|| format!("shmctl dump-cache: --holds takes a count from 1 to {MAX_HOLDS}, got {k:?}")),
        _ => Err("usage: neural-forge-cli shmctl dump-cache [--holds K]".into()),
    }
}

fn cmd_dump_cache(header: &ShmHeader, args: &[String]) -> bool {
    match dump_cache_holds(args) {
        Ok(k) => {
            header.preupscale_dump_request.store(k, Ordering::Relaxed);
            println!("preupscale_dump_request={k} -- the game's next {k} model-mode holds go to a new $XDG_DATA_HOME/neural-forge/captures/preupscale-cache-<ms>/");
            true
        }
        Err(e) => {
            eprintln!("{e}");
            false
        }
    }
}

/// Checks the argument count of a `shmctl` command line before anything is opened, so an extra
/// or missing argument is a usage error. `capture`'s own options are checked by `cmd_capture`.
pub fn check_args(args: &[String]) -> Result<(), String> {
    let n = args.len();
    match args.first().map(String::as_str) {
        None => Err("shmctl takes a subcommand".into()),
        Some("status" | "reset") if n == 1 => Ok(()),
        Some("set") if n == 3 => Ok(()),
        Some("toggle") if n == 2 => Ok(()),
        Some("capture") => capture_args(&args[1..]).map(|_| ()),
        Some("dump-cache") => dump_cache_holds(&args[1..]).map(|_| ()),
        Some(sub @ ("status" | "reset" | "set" | "toggle")) => Err(format!("wrong number of arguments for shmctl {sub}")),
        Some(other) => Err(format!("unknown shmctl subcommand {other:?}")),
    }
}

pub fn run(args: &[String]) -> std::process::ExitCode {
    let cfg = neural_forge_supervisor::Config::load();
    let mapping = match neural_forge_supervisor::open_channel(&cfg) {
        Ok(mapping) => mapping,
        Err(e) => {
            eprintln!("shmctl: {e}: {}", neural_forge_supervisor::channel_path(&cfg));
            return std::process::ExitCode::FAILURE;
        }
    };
    neural_forge_supervisor::apply_saved_settings(&cfg, &mapping);
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
        Some("capture") => cmd_capture(header, &args[1..]),
        Some("dump-cache") => cmd_dump_cache(header, &args[1..]),
        Some("reset") => {
            header.reset_persisted_settings();
            println!("settings reset to defaults; layer session preserved");
            true
        }
        _ => {
            usage();
            false
        }
    };
    // A changed setting is saved the way the GUI and `profile load` save one, so it survives a
    // reboot and the next cold start does not put the old value back.
    let changes_settings = matches!(args.first().map(String::as_str), Some("set" | "toggle" | "reset"));
    if ok && changes_settings {
        if let Err(e) = neural_forge_supervisor::save_settings(header) {
            eprintln!("shmctl: applied to the live channel, but saving config.ini failed: {e}");
            return std::process::ExitCode::FAILURE;
        }
    }
    if ok {
        std::process::ExitCode::SUCCESS
    } else {
        std::process::ExitCode::FAILURE
    }
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
    fn dump_cache_takes_an_optional_hold_count() {
        let a = |s: &str| s.split_whitespace().map(String::from).collect::<Vec<_>>();
        assert_eq!(dump_cache_holds(&a("")), Ok(16));
        assert_eq!(dump_cache_holds(&a("--holds 8")), Ok(8));
        assert_eq!(dump_cache_holds(&a("--holds 64")), Ok(64));
        assert!(dump_cache_holds(&a("--holds 65")).is_err());
        assert!(dump_cache_holds(&a("--holds 0")).is_err());
        assert!(dump_cache_holds(&a("--holds")).is_err());
        assert!(dump_cache_holds(&a("8")).is_err());
        assert!(check_args(&a("dump-cache --holds 4")).is_ok());
        assert!(check_args(&a("dump-cache --holds 100")).is_err());
        let header = ShmHeader::default();
        assert!(cmd_dump_cache(&header, &a("--holds 4")));
        assert_eq!(header.preupscale_dump_request.load(Ordering::Relaxed), 4);
    }

    #[test]
    fn resolve_finds_debug_view_via_persisted_settings() {
        // debug_view/apply_model/compare_mode/hold_frame are persisted settings.
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
    fn store_marks_an_untouched_header_as_configured() {
        let header = ShmHeader::default();
        header.init_defaults();
        assert_eq!(header.tuning_seq.load(Ordering::Relaxed), 0);
        assert!(cmd_set(&header, "intensity", "0.5"));
        assert_ne!(header.tuning_seq.load(Ordering::Relaxed), 0, "the next start() must not overwrite this from config.ini");
        assert_ne!(header.control_seq.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn capture_arguments_are_bounded() {
        let a = |s: &str| s.split_whitespace().map(String::from).collect::<Vec<_>>();
        let max = neural_forge_protocol::capture::MAX_FRAMES;
        assert_eq!(capture_args(&a("")), Ok(CaptureArgs::Take { frames: 1, view: None }));
        assert_eq!(capture_args(&a("--frames 150 --view 6")), Ok(CaptureArgs::Take { frames: 150, view: Some(6) }));
        assert_eq!(capture_args(&a(&format!("--frames {max}"))), Ok(CaptureArgs::Take { frames: max, view: None }));
        for bad in ["--frames 0", &format!("--frames {}", max + 1), "--frames", "--view 7", "--view x", "3", "--frames 2 --frames 3", "--cancel --frames 2"] {
            assert!(capture_args(&a(bad)).is_err(), "{bad}");
            assert!(check_args(&a(&format!("capture {bad}"))).is_err(), "{bad}");
        }
        assert_eq!(capture_args(&a("--cancel")), Ok(CaptureArgs::Cancel));
        // A refused view stores nothing.
        let header = ShmHeader::default();
        header.init_defaults();
        assert!(!cmd_capture(&header, &a("--view 9")));
        assert_eq!(header.debug_view.load(Ordering::Relaxed), 0);
        assert_eq!(header.capture_request.load(Ordering::Relaxed), 0, "and asks for nothing");
    }

    /// The request goes out with its own id, the layer's outcome for that id is what is reported, and
    /// a request no game takes is withdrawn rather than left to fire on a game started later.
    #[test]
    fn capture_waits_for_its_own_outcome_and_withdraws_an_untaken_request() {
        use neural_forge_protocol::capture;
        let header = std::sync::Arc::new(ShmHeader::default());
        let id = capture::request(&header, 2).unwrap();
        let layer = {
            let header = header.clone();
            std::thread::spawn(move || {
                // Stands in for the game's layer: takes the request, publishes another id's outcome
                // first (ignored), then this one's.
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
                let request = loop {
                    if let Some(r) = capture::take(&header, |_| true) {
                        break r;
                    }
                    assert!(std::time::Instant::now() < deadline);
                    std::thread::sleep(std::time::Duration::from_millis(1));
                };
                capture::publish(&header, &capture::Outcome { id: request.id + 1, state: capture::state::WRITTEN, ..Default::default() });
                std::thread::sleep(std::time::Duration::from_millis(30));
                capture::publish(&header, &capture::Outcome { id: request.id, state: capture::state::WRITTEN, done: 2, total: 2, composed: 2, paths: "/c/series-1".into(), ..Default::default() });
            })
        };
        let outcome = wait_for_capture(&header, id, std::time::Duration::from_secs(5), std::time::Duration::from_secs(10)).unwrap();
        layer.join().unwrap();
        assert_eq!((outcome.id, outcome.done, outcome.paths.as_str()), (id, 2, "/c/series-1"));

        let id = capture::request(&header, 1).unwrap();
        let err = wait_for_capture(&header, id, std::time::Duration::from_millis(50), std::time::Duration::from_secs(10)).unwrap_err();
        assert!(err.contains("no game took"), "{err}");
        assert_eq!(capture::pending(&header), None, "withdrawn: nothing waits to fire on the next game");
    }

    #[test]
    fn capture_cancel_withdraws_a_waiting_request() {
        let header = ShmHeader::default();
        neural_forge_protocol::capture::request(&header, 3).unwrap();
        assert!(cmd_capture(&header, &["--cancel".to_string()]));
        assert_eq!(header.capture_request.load(Ordering::Relaxed), 0);
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
    fn cmd_set_rejects_non_finite_negative_and_out_of_range_values() {
        let header = ShmHeader::default();
        header.init_defaults();
        for bad in ["nan", "inf", "-inf", "-0.5", "4.5"] {
            assert!(!cmd_set(&header, "intensity", bad), "{bad}");
        }
        assert_eq!(f32::from_bits(header.intensity_bits.load(Ordering::Relaxed)), 1.0);
        assert!(!cmd_set(&header, "model_interval", "0"));
        assert!(!cmd_set(&header, "style", "-1"));
        assert!(!cmd_set(&header, "debug_view", "7"));
        assert_eq!(header.tuning_seq.load(Ordering::Relaxed), 0, "a refused value changes nothing");
        assert!(cmd_set(&header, "intensity", "4"));
        assert_eq!(f32::from_bits(header.intensity_bits.load(Ordering::Relaxed)), 4.0);
    }
}
