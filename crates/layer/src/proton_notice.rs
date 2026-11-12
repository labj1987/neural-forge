//! One line in the game's log when it runs with a Neural Forge Proton copy older than the Proton it was made from
//! (docs/PROTON_NGX.md). The layer is the only Neural Forge code that runs at a game's launch (the launch options
//! are environment variables), so it finds the Proton the process runs with from what Steam gives every game
//! (`STEAM_COMPAT_TOOL_PATHS`, the compat tool folders) and, whatever environment reached the process, from where
//! Wine's own libraries are mapped (`/proc/self/maps`). The comparison is the supervisor's own (`neural_forge_protocol::proton_copy`): the
//! copy's marker names its source and the source's version then.

use std::sync::Once;

/// Logs the notice for each stale copy among `STEAM_COMPAT_TOOL_PATHS`, once per process.
pub(crate) fn note_once() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let paths = std::env::var("STEAM_COMPAT_TOOL_PATHS").ok();
        let maps = std::fs::read_to_string("/proc/self/maps").unwrap_or_default();
        for line in neural_forge_protocol::proton_copy::launch_notices(paths.as_deref(), &maps) {
            crate::log!("{line}");
        }
    });
}
