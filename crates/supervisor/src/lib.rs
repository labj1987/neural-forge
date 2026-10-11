//! Shared front-end support for `neural-forge-cli` and `neural-forge`: config, XDG paths, the
//! settings channel, installation, the model extraction and the diagnosis (`doctor`), so both
//! front ends do each of these the same way. Linux-only: reads XDG env vars.

pub mod config;
pub mod doctor;
pub mod install;
pub mod launch_options;
pub mod model;
pub mod model_shape;
pub mod paths;
pub mod profiles;
pub mod steam;
pub mod vdf;

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

/// Applies the saved settings (`config.ini`) to `mapping`, the channel just opened with
/// [`open_channel`], if nothing has applied them since it was last initialised. The header is
/// initialised with defaults by whichever of the layer, the GUI or the CLI finds it missing; the
/// layer applies the saved settings itself then (`neural_forge_protocol::persist::apply_saved`),
/// and the GUI and CLI call this right after opening the channel for the same reason, so a
/// command run on a cold boot acts on the saved settings, not the defaults. `init_defaults`
/// leaves `tuning_seq` at 0 and `persist::apply` bumps it, which is what "not yet applied" is
/// read from. Returns whether it applied them.
pub fn apply_saved_settings(cfg: &Config, mapping: &neural_forge_protocol::mapping::Mapping) -> bool {
    let hdr = mapping.header();
    if hdr.tuning_seq.load(std::sync::atomic::Ordering::Relaxed) != 0 {
        return false;
    }
    neural_forge_protocol::persist::apply(hdr, &cfg.settings);
    hdr.control_seq.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    true
}

/// Writes the live header's settings to `config.ini`, so a change made on the channel also
/// survives a reboot.
pub fn save_settings(header: &neural_forge_protocol::ShmHeader) -> std::io::Result<()> {
    let mut cfg = Config::load();
    cfg.replace_tuning(neural_forge_protocol::persist::snapshot(header));
    cfg.save()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saved_settings_are_applied_to_a_fresh_header_once() {
        let dir = std::env::temp_dir().join(format!("neural-forge-apply-saved-{}", std::process::id()));
        let mapping = neural_forge_protocol::mapping::open_path(&dir.join("shm.bin").to_string_lossy()).unwrap();
        let mut cfg = Config::default();
        cfg.settings.insert("set_intensity".into(), "2.5".into());
        let intensity = || f32::from_bits(mapping.header().intensity_bits.load(std::sync::atomic::Ordering::Relaxed));
        assert_eq!(intensity(), 1.0, "a fresh header holds the defaults");
        assert!(apply_saved_settings(&cfg, &mapping));
        assert_eq!(intensity(), 2.5);
        // Applied once: a later change on the live header is not overwritten.
        mapping.header().intensity_bits.store(0.5f32.to_bits(), std::sync::atomic::Ordering::Relaxed);
        assert!(!apply_saved_settings(&cfg, &mapping));
        assert_eq!(intensity(), 0.5);
        drop(mapping);
        std::fs::remove_dir_all(&dir).ok();
    }

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
