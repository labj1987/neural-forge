//! `neural-forge-cli` -- the command-line front end: config, the NVIDIA DLL import and the model
//! extraction, install/uninstall, the live settings channel (`shmctl`) and settings profiles. Shares
//! its logic with the GUI through `neural_forge_supervisor` and `neural_forge_protocol`.

mod shmctl;

use std::process::ExitCode;

use neural_forge_supervisor::{paths, Config};

fn usage() {
    eprintln!(
        "usage: neural-forge-cli <command>\n\n\
         commands:\n\
         \x20 init                 create the default config\n\
         \x20 status               show the config, channel and model\n\
         \x20 doctor               check the config, NVIDIA DLL, model and paths\n\
         \x20 config               print the effective config\n\
         \x20 import-binaries DIR  copy NVIDIA's nvngx_dlssnr.dll into the user data dir\n\
         \x20 extract-model DIR    write the model directory from nvngx_dlssnr.dll\n\
         \x20                     (DIR holds it, or is the DLL) into the user data dir\n\
         \x20 install --appdir DIR install an extracted AppImage AppDir into\n\
         \x20                     persistent user storage (see scripts/install.py --\n\
         \x20                     same operation, same record, either tool works)\n\
         \x20 uninstall [--purge]  remove unchanged tracked installed files; --purge also\n\
         \x20                     removes config, data (the DLL, the model), state and\n\
         \x20                     /tmp/neural-forge-$UID\n\
         \x20 shmctl <sub>         raw status/set/toggle/capture against a running\n\
         \x20                     instance's live SHM header (see `shmctl help`)\n\
         \x20 profile <sub>        save/load/list/delete named settings profiles\n\
         \x20                     (see `profile help`)"
    );
}

fn profile_usage() {
    eprintln!(
        "usage: neural-forge-cli profile <list|save|load|delete>\n\n\
         \x20 list          print every saved profile name\n\
         \x20 save <name>   snapshot the running instance's current settings as <name>\n\
         \x20 load <name>   apply <name>'s settings to the running instance and persist\n\
         \x20               them to config.ini\n\
         \x20 delete <name> remove a saved profile\n\n\
         Profiles live in $XDG_CONFIG_HOME/neural-forge/profiles.ini. `save`/`load`\n\
         attach to the live SHM mapping the same way `shmctl` does (config.ini's\n\
         shm=, else $NEURAL_FORGE_SHM, else the default under /tmp/neural-forge-$UID)."
    );
}

fn cmd_profile_list() -> ExitCode {
    let profiles = neural_forge_supervisor::profiles::load_all();
    if profiles.is_empty() {
        println!("no saved profiles ({})", neural_forge_supervisor::profiles::profiles_file());
        return ExitCode::SUCCESS;
    }
    for name in profiles.keys() {
        println!("{name}");
    }
    ExitCode::SUCCESS
}

fn cmd_profile_save(name: &str) -> ExitCode {
    let channel_cfg = Config::load();
    let mapping = match neural_forge_supervisor::open_channel(&channel_cfg) {
        Ok(mapping) => mapping,
        Err(e) => {
            eprintln!("profile save: {e}: {}", neural_forge_supervisor::channel_path(&channel_cfg));
            return ExitCode::FAILURE;
        }
    };
    let settings = neural_forge_protocol::persist::snapshot(mapping.header());
    match neural_forge_supervisor::profiles::save_profile(name, settings) {
        Ok(()) => {
            println!("saved profile {name:?} to {}", neural_forge_supervisor::profiles::profiles_file());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("profile save: failed to write {}: {e}", neural_forge_supervisor::profiles::profiles_file());
            ExitCode::FAILURE
        }
    }
}

fn cmd_profile_load(name: &str) -> ExitCode {
    let profiles = neural_forge_supervisor::profiles::load_all();
    let Some(settings) = profiles.get(name) else {
        eprintln!("profile load: no such profile {name:?} (see `profile list`)");
        return ExitCode::FAILURE;
    };
    let channel_cfg = Config::load();
    let mapping = match neural_forge_supervisor::open_channel(&channel_cfg) {
        Ok(mapping) => mapping,
        Err(e) => {
            eprintln!("profile load: {e}: {}", neural_forge_supervisor::channel_path(&channel_cfg));
            return ExitCode::FAILURE;
        }
    };
    let header = mapping.header();
    neural_forge_protocol::persist::apply(header, settings);
    // Matches the GUI's own reset-settings flow: applying to the live header alone
    // only affects the running session, so also fold the new values into config.ini
    // via a fresh snapshot (picks up every persisted setting, not just what this
    // profile happened to list) so the change survives a reboot too.
    let mut cfg = Config::load();
    cfg.replace_tuning(neural_forge_protocol::persist::snapshot(header));
    if let Err(e) = cfg.save() {
        eprintln!("profile load: applied to the running instance, but saving config.ini failed: {e}");
        return ExitCode::FAILURE;
    }
    println!("loaded profile {name:?}");
    ExitCode::SUCCESS
}

fn cmd_profile_delete(name: &str) -> ExitCode {
    match neural_forge_supervisor::profiles::delete_profile(name) {
        Ok(true) => {
            println!("deleted profile {name:?}");
            ExitCode::SUCCESS
        }
        Ok(false) => {
            eprintln!("profile delete: no such profile {name:?} (see `profile list`)");
            ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!("profile delete: failed to write {}: {e}", neural_forge_supervisor::profiles::profiles_file());
            ExitCode::FAILURE
        }
    }
}

fn cmd_profile(args: &[String]) -> ExitCode {
    match (args.first().map(String::as_str), args.get(1)) {
        (Some("list"), _) => cmd_profile_list(),
        (Some("save"), Some(name)) => cmd_profile_save(name),
        (Some("load"), Some(name)) => cmd_profile_load(name),
        (Some("delete"), Some(name)) => cmd_profile_delete(name),
        (Some("help"), _) | (None, _) => {
            profile_usage();
            ExitCode::SUCCESS
        }
        _ => {
            profile_usage();
            ExitCode::FAILURE
        }
    }
}

fn default_config() -> Config {
    // `shm` stays unset: the channel then follows $NEURAL_FORGE_SHM or the default (see
    // `neural_forge_supervisor::channel_path`) rather than pinning today's default for good.
    Config { binaries: paths::binaries_dir(), ..Config::default() }
}

fn cmd_init() -> ExitCode {
    if std::path::Path::new(&paths::config_file()).exists() {
        println!("config already exists: {}", paths::config_file());
        return ExitCode::SUCCESS;
    }
    match default_config().save() {
        Ok(()) => {
            println!("wrote {}", paths::config_file());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("failed to write config: {e}");
            ExitCode::FAILURE
        }
    }
}

fn cmd_config() -> ExitCode {
    let cfg = Config::load();
    println!("config_file={}", paths::config_file());
    println!("data_dir={}", paths::data_dir());
    println!("state_dir={}", paths::state_dir());
    println!("binaries={}", cfg.binaries);
    println!("shm={}", neural_forge_supervisor::channel_path(&cfg));
    println!("model_dir={}", neural_forge_supervisor::model::model_dir());
    for (k, v) in &cfg.settings {
        println!("{k}={v}");
    }
    ExitCode::SUCCESS
}

/// The extracted model's build, if one is installed.
fn model_build() -> Option<String> {
    neural_forge_supervisor::model::installed(std::path::Path::new(&neural_forge_supervisor::model::model_dir()))
}

/// ", not verified against NVIDIA's runtime" when the installed model came from such a build.
fn unverified_note() -> &'static str {
    let dir = neural_forge_supervisor::model::model_dir();
    match neural_forge_supervisor::model::installed_verified(std::path::Path::new(&dir)) {
        Some(false) => ", not verified against NVIDIA's runtime",
        _ => "",
    }
}

fn cmd_status() -> ExitCode {
    println!("config: {}", paths::config_file());
    println!("channel: {}", neural_forge_supervisor::channel_path(&Config::load()));
    match model_build() {
        Some(build) => println!("model: build {build}{} in {}", unverified_note(), neural_forge_supervisor::model::model_dir()),
        None => println!("model: not extracted (see `neural-forge-cli extract-model DIR`)"),
    }
    println!("state: {}", paths::state_dir());
    ExitCode::SUCCESS
}

fn cmd_doctor() -> ExitCode {
    let mut ok = true;
    let cfg = Config::load();

    print!("config: {}\n  ", paths::config_file());
    if std::path::Path::new(&paths::config_file()).exists() {
        println!("ok");
    } else {
        println!("missing (run `neural-forge-cli init`)");
        ok = false;
    }

    let binaries = if cfg.binaries.is_empty() { paths::binaries_dir() } else { cfg.binaries.clone() };
    let ngx_dll = std::path::Path::new(&binaries).join("nvngx_dlssnr.dll");
    print!("binaries: {binaries}\n  nvngx_dlssnr.dll: ");
    if ngx_dll.exists() {
        println!("ok");
    } else {
        println!("missing (only needed to extract the model; see `neural-forge-cli import-binaries DIR`)");
    }

    print!("model: {}\n  ", neural_forge_supervisor::model::model_dir());
    match model_build() {
        Some(build) => println!("ok (build {build}{})", unverified_note()),
        None => {
            println!("missing (run `neural-forge-cli extract-model DIR`)");
            ok = false;
        }
    }

    print!("runtime dir: {}\n  ", neural_forge_protocol::shm_runtime_dir());
    match paths::ensure_dirs() {
        Ok(()) => println!("ok"),
        Err(e) => {
            println!("not writable: {e}");
            ok = false;
        }
    }

    if ok { ExitCode::SUCCESS } else { ExitCode::FAILURE }
}

fn cmd_install(appdir: Option<&String>) -> ExitCode {
    let Some(appdir) = appdir else {
        eprintln!("usage: neural-forge-cli install --appdir DIR");
        return ExitCode::FAILURE;
    };
    match neural_forge_supervisor::install::install(std::path::Path::new(appdir)) {
        Ok(report) => {
            println!("Installed Neural Forge. CLI: {}", report.cli_path.display());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("install failed: {e}");
            ExitCode::FAILURE
        }
    }
}

fn cmd_uninstall(purge: bool) -> ExitCode {
    if purge {
        return match neural_forge_supervisor::install::purge() {
            Ok(removed) => {
                for path in &removed {
                    println!("removed {}", path.display());
                }
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("purge failed: {e}");
                ExitCode::FAILURE
            }
        };
    }
    match neural_forge_supervisor::install::uninstall() {
        Ok(preserved) => {
            for path in &preserved {
                println!("preserved changed/missing file: {}", path.display());
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("uninstall failed: {e}");
            ExitCode::FAILURE
        }
    }
}

fn cmd_import_binaries(dir: Option<&String>) -> ExitCode {
    let Some(dir) = dir else {
        eprintln!("usage: neural-forge-cli import-binaries DIR");
        return ExitCode::FAILURE;
    };
    let src = std::path::Path::new(dir);
    if !src.is_dir() {
        eprintln!("not a directory: {dir}");
        return ExitCode::FAILURE;
    }
    let dest = paths::binaries_dir();
    if let Err(e) = std::fs::create_dir_all(&dest) {
        eprintln!("failed to create {dest}: {e}");
        return ExitCode::FAILURE;
    }
    // Only `nvngx_dlssnr.dll`: the model's weights are extracted from it (`extract-model`).
    let name = "nvngx_dlssnr.dll";
    let from = src.join(name);
    let copied = if from.is_file() {
        if let Err(e) = std::fs::copy(&from, std::path::Path::new(&dest).join(name)) {
            eprintln!("failed to copy {name}: {e}");
            return ExitCode::FAILURE;
        }
        1
    } else {
        0
    };
    println!("imported {copied} file(s) to {dest}");
    ExitCode::SUCCESS
}

fn cmd_extract_model(source: Option<&String>) -> ExitCode {
    let Some(source) = source else {
        eprintln!("usage: neural-forge-cli extract-model DIR");
        return ExitCode::FAILURE;
    };
    let out = neural_forge_supervisor::model::model_dir();
    match neural_forge_supervisor::model::extract(std::path::Path::new(source), std::path::Path::new(&out)) {
        Ok(done) => {
            println!(
                "extracted build {} ({} tensors, {} bytes) to {}",
                done.build,
                done.tensors,
                done.bytes,
                done.dir.display()
            );
            if !done.verified {
                let v = neural_forge_supervisor::model::VERIFIED_BUILDS[0];
                println!(
                    "Build {} has the same network as the verified build {}.{}.{}, but its output has not been compared with NVIDIA's runtime.",
                    done.build, v[0], v[1], v[2]
                );
            }
            ExitCode::SUCCESS
        }
        Err(neural_forge_supervisor::model::ModelError::Shape(report)) => {
            eprintln!("extract-model refused the DLL:\n{report}");
            ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!("extract-model failed: {e}");
            ExitCode::FAILURE
        }
    }
}

fn main() -> ExitCode {
    // `args_os`: `args()` panics on an argument that is not valid UTF-8 (a path, say).
    let args: Vec<String> = std::env::args_os().map(|a| a.to_string_lossy().into_owned()).collect();
    let Some(command) = args.get(1) else {
        usage();
        return ExitCode::FAILURE;
    };

    match command.as_str() {
        "init" => cmd_init(),
        "status" => cmd_status(),
        "doctor" => cmd_doctor(),
        "config" => cmd_config(),
        "import-binaries" => cmd_import_binaries(args.get(2)),
        "extract-model" => cmd_extract_model(args.get(2)),
        "install" => {
            let appdir = args.iter().position(|a| a == "--appdir").and_then(|i| args.get(i + 1));
            cmd_install(appdir)
        }
        "uninstall" => cmd_uninstall(args.get(2).map(String::as_str) == Some("--purge")),
        "shmctl" => shmctl::run(&args[2..]),
        "profile" => cmd_profile(&args[2..]),
        "help" | "--help" | "-h" => {
            usage();
            ExitCode::SUCCESS
        }
        other => {
            eprintln!("unknown command: {other}");
            usage();
            ExitCode::FAILURE
        }
    }
}
