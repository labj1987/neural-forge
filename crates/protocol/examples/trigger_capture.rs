//! Asks an already-running game's layer for a capture (`neural_forge_protocol::capture::request`),
//! so its next present writes a matched before/after PNG pair to
//! `$XDG_DATA_HOME/neural-forge/captures` (see `neural_forge_layer::series`). The same request the
//! GUI's capture button and `neural-forge-cli shmctl capture` make, without waiting for the outcome.
//!
//! Respects `$NEURAL_FORGE_SHM`/`$NEURAL_FORGE_UID` the same way every other tool in this
//! workspace does -- point it at a specific instance's mapping the same way you'd
//! point `vkcube`/the layer at it.

fn main() {
    let mapping = match neural_forge_protocol::mapping::open() {
        Ok(mapping) => mapping,
        Err(e) => {
            eprintln!("trigger_capture: {e} (see $NEURAL_FORGE_SHM/$NEURAL_FORGE_UID)");
            std::process::exit(1);
        }
    };
    // Optional `debug_view` override (`SETTING_BOUNDS`), set before the request so the very next
    // present both shows and captures that view.
    if let Some(arg) = std::env::args().nth(1) {
        let (min, max) = neural_forge_protocol::setting_bounds("debug_view").unwrap_or((0.0, 0.0));
        match arg.parse::<u32>().ok().filter(|&v| (min..=max).contains(&(v as f32))) {
            Some(view) => {
                mapping.header().debug_view.store(view, std::sync::atomic::Ordering::Relaxed);
                mapping.header().control_seq.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                println!("trigger_capture: debug_view set to {view}");
            }
            None => {
                eprintln!("trigger_capture: the debug view is {min} to {max}, got {arg:?}");
                std::process::exit(2);
            }
        }
    }
    match neural_forge_protocol::capture::request(mapping.header(), 1) {
        Ok(id) => println!("trigger_capture: capture {id} requested -- `neural-forge-cli shmctl status` shows what the layer wrote"),
        Err(e) => {
            eprintln!("trigger_capture: {e}");
            std::process::exit(1);
        }
    }
}
