//! Shared front-end support for `neural-forge-cli` and `neural-forge`: config, XDG paths, the
//! settings channel, installation and the model extraction, so both front ends do each of these the
//! same way. Linux-only: reads XDG env vars.

pub mod config;
pub mod install;
pub mod model;
pub mod model_shape;
pub mod paths;
pub mod profiles;

pub use config::Config;

/// The one channel path everything this crate touches uses: `config.ini`'s `shm=` when it is set (and
/// inside this project's namespace), else `$NEURAL_FORGE_SHM`, else the default under the runtime
/// directory. Saved settings are applied to it, and the GUI and CLI open it through [`open_channel`].
pub fn channel_path(cfg: &Config) -> String {
    if !cfg.shm.is_empty() && neural_forge_protocol::isolated_path(&cfg.shm) {
        return cfg.shm.clone();
    }
    neural_forge_protocol::env::var("NEURAL_FORGE_SHM")
        .filter(|s| !s.is_empty() && neural_forge_protocol::isolated_path(s))
        .unwrap_or_else(neural_forge_protocol::shm_default_path)
}

/// Opens the mapping at [`channel_path`].
pub fn open_channel(cfg: &Config) -> Result<neural_forge_protocol::mapping::Mapping, neural_forge_protocol::mapping::OpenError> {
    neural_forge_protocol::mapping::open_path(&channel_path(cfg))
}

/// Applies the saved settings (`config.ini`) to the live header at [`channel_path`] if nothing has
/// applied them since it was last initialised. The header is initialised with defaults by whichever of
/// the layer or the GUI finds it missing; the layer applies the saved settings itself then
/// (`neural_forge_protocol::persist::apply_saved`), and the GUI calls this for the same reason.
/// `init_defaults` leaves `tuning_seq` at 0 and `persist::apply` bumps it, which is what "not yet
/// applied" is read from.
pub fn apply_saved_settings(cfg: &Config) {
    let Ok(mapping) = open_channel(cfg) else { return };
    let hdr = mapping.header();
    if hdr.tuning_seq.load(std::sync::atomic::Ordering::Relaxed) == 0 {
        neural_forge_protocol::persist::apply(hdr, &cfg.settings);
        hdr.control_seq.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_path_prefers_config_then_environment_then_default() {
        let _guard = crate::paths::tests::XDG_DATA_HOME_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let prev = std::env::var("NEURAL_FORGE_SHM").ok();
        let mut cfg = Config { shm: "/tmp/neural-forge-x/configured.bin".into(), ..Config::default() };
        std::env::set_var("NEURAL_FORGE_SHM", "/tmp/neural-forge-x/env.bin");
        assert_eq!(channel_path(&cfg), "/tmp/neural-forge-x/configured.bin");
        cfg.shm.clear();
        assert_eq!(channel_path(&cfg), "/tmp/neural-forge-x/env.bin");
        std::env::set_var("NEURAL_FORGE_SHM", "/tmp/dlssnr-1000/shm.bin");
        assert_eq!(channel_path(&cfg), neural_forge_protocol::shm_default_path(), "a path outside the namespace is ignored");
        std::env::remove_var("NEURAL_FORGE_SHM");
        assert_eq!(channel_path(&cfg), neural_forge_protocol::shm_default_path());
        match prev {
            Some(v) => std::env::set_var("NEURAL_FORGE_SHM", v),
            None => std::env::remove_var("NEURAL_FORGE_SHM"),
        }
    }
}
