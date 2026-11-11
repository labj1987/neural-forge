//! One line in the game's log when it runs with a Neural Forge Proton copy older than the Proton it was made from
//! (docs/PROTON_NGX.md). The layer is the only Neural Forge code that runs at a game's launch (the launch options
//! are environment variables), so it reads what Steam gives every game: `STEAM_COMPAT_TOOL_PATHS`, the compat tool
//! folders the game runs with. The comparison is the supervisor's own (`neural_forge_protocol::proton_copy`): the
//! copy's marker names its source and the source's version then.

use std::sync::Once;

/// Logs the notice for each stale copy among `STEAM_COMPAT_TOOL_PATHS`, once per process.
pub(crate) fn note_once() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        if let Ok(paths) = std::env::var("STEAM_COMPAT_TOOL_PATHS") {
            for line in neural_forge_protocol::proton_copy::stale_notices(&paths) {
                crate::log!("{line}");
            }
        }
    });
}
