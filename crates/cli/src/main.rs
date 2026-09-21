//! `neuralforge-cli` — the helper-manager: init/setup/start/stop/restart/status/doctor/
//! config/runners/detect-gpu/import-binaries. Replaces upstream's ~900-line bash
//! `neuralforge-helper` script with the same command surface (a CLI contract, not
//! upstream's expression of it) reimplemented in Rust, sharing logic with the GUI
//! through `neuralforge_protocol` instead of duplicating it in shell.

mod gpu;
mod shmctl;

use std::process::ExitCode;
use std::time::Duration;

use neuralforge_supervisor::{install_dir, paths, Config};

fn usage() {
    eprintln!(
        "usage: neuralforge-cli <command>\n\n\
         commands:\n\
         \x20 init                 create default config\n\
         \x20 setup                init config, dxvk config, and managed prefix if needed\n\
         \x20 start                start helper using configured runner\n\
         \x20 stop                 stop helper process tree\n\
         \x20 restart              stop then start\n\
         \x20 status               show helper status\n\
         \x20 doctor               check runner, prefix, dxvk, binaries, and paths\n\
         \x20 config               print effective config\n\
         \x20 runners              list discovered custom compatibility tool runners\n\
         \x20 detect-gpu           print detected NVIDIA PCI vendor/device\n\
         \x20 import-binaries DIR  copy NVIDIA NGX DLLs into user data dir\n\
         \x20 install --appdir DIR install an extracted AppImage AppDir into\n\
         \x20                     persistent user storage (see scripts/install.py --\n\
         \x20                     same operation, same record, either tool works)\n\
         \x20 uninstall            remove only unchanged tracked installed files\n\
         \x20 shmctl <sub>         raw status/set/toggle/capture against a running\n\
         \x20                     instance's live SHM header (see `shmctl help`)\n\
         \x20 profile <sub>        save/load/list/delete named settings profiles\n\
         \x20                     (see `profile help`)"
    );
}

fn profile_usage() {
    eprintln!(
        "usage: neuralforge-cli profile <list|save|load|delete>\n\n\
         \x20 list          print every saved profile name\n\
         \x20 save <name>   snapshot the running instance's current settings as <name>\n\
         \x20 load <name>   apply <name>'s settings to the running instance and persist\n\
         \x20               them to config.ini\n\
         \x20 delete <name> remove a saved profile\n\n\
         Profiles live in $XDG_CONFIG_HOME/neuralforge/profiles.ini. `save`/`load`\n\
         attach to the live SHM mapping the same way `shmctl` does (see\n\
         $NEURALFORGE_SHM/$NEURALFORGE_UID)."
    );
}

fn cmd_profile_list() -> ExitCode {
    let profiles = neuralforge_supervisor::profiles::load_all();
    if profiles.is_empty() {
        println!("no saved profiles ({})", neuralforge_supervisor::profiles::profiles_file());
        return ExitCode::SUCCESS;
    }
    for name in profiles.keys() {
        println!("{name}");
    }
    ExitCode::SUCCESS
}

fn cmd_profile_save(name: &str) -> ExitCode {
    let Some(mapping) = neuralforge_protocol::mapping::open() else {
        eprintln!("profile save: failed to open the SHM mapping (see $NEURALFORGE_SHM/$NEURALFORGE_UID)");
        return ExitCode::FAILURE;
    };
    let settings = neuralforge_protocol::persist::snapshot(mapping.header());
    match neuralforge_supervisor::profiles::save_profile(name, settings) {
        Ok(()) => {
            println!("saved profile {name:?} to {}", neuralforge_supervisor::profiles::profiles_file());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("profile save: failed to write {}: {e}", neuralforge_supervisor::profiles::profiles_file());
            ExitCode::FAILURE
        }
    }
}

fn cmd_profile_load(name: &str) -> ExitCode {
    let profiles = neuralforge_supervisor::profiles::load_all();
    let Some(settings) = profiles.get(name) else {
        eprintln!("profile load: no such profile {name:?} (see `profile list`)");
        return ExitCode::FAILURE;
    };
    let Some(mapping) = neuralforge_protocol::mapping::open() else {
        eprintln!("profile load: failed to open the SHM mapping (see $NEURALFORGE_SHM/$NEURALFORGE_UID)");
        return ExitCode::FAILURE;
    };
    let header = mapping.header();
    neuralforge_protocol::persist::apply(header, settings);
    // Matches the GUI's own reset-settings flow: applying to the live header alone
    // only affects the running session, so also fold the new values into config.ini
    // via a fresh snapshot (picks up every persisted setting, not just what this
    // profile happened to list) so the change survives a reboot too.
    let mut cfg = Config::load();
    cfg.settings = neuralforge_protocol::persist::snapshot(header);
    if let Err(e) = cfg.save() {
        eprintln!("profile load: applied to the running instance, but saving config.ini failed: {e}");
        return ExitCode::FAILURE;
    }
    println!("loaded profile {name:?}");
    ExitCode::SUCCESS
}

fn cmd_profile_delete(name: &str) -> ExitCode {
    match neuralforge_supervisor::profiles::delete_profile(name) {
        Ok(true) => {
            println!("deleted profile {name:?}");
            ExitCode::SUCCESS
        }
        Ok(false) => {
            eprintln!("profile delete: no such profile {name:?} (see `profile list`)");
            ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!("profile delete: failed to write {}: {e}", neuralforge_supervisor::profiles::profiles_file());
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
    let mut cfg = Config::default();
    if let Some(proton) = neuralforge_supervisor::runners::best_proton() {
        cfg.runner_type = "proton".to_string();
        cfg.runner_path = proton.path.to_string_lossy().into_owned();
    } else if let Some(wine) = neuralforge_supervisor::runners::find_wine() {
        cfg.runner_type = "wine".to_string();
        cfg.runner_path = wine.to_string_lossy().into_owned();
    } else {
        cfg.runner_type = "custom".to_string();
    }
    cfg.binaries = paths::binaries_dir();
    if let Some((vendor, device)) = gpu::detect_nvidia_gpu() {
        cfg.dxvk_vendor = format!("{vendor:04x}");
        cfg.dxvk_device = format!("{device:04x}");
    }
    cfg.shm = neuralforge_protocol::shm_default_path();
    cfg.log = paths::log_file();
    cfg
}

fn cmd_init() -> ExitCode {
    if std::path::Path::new(&paths::config_file()).exists() {
        println!("config already exists: {}", paths::config_file());
        return ExitCode::SUCCESS;
    }
    let cfg = default_config();
    match cfg.save() {
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
    println!("prefix_dir={}", paths::prefix_dir());
    println!("runner_type={}", cfg.runner_type);
    println!("runner_path={}", cfg.runner_path);
    println!("binaries={}", cfg.binaries);
    println!("shm={}", cfg.shm);
    println!("log={}", cfg.log);
    println!("dxvk_vendor={}", cfg.dxvk_vendor);
    println!("dxvk_device={}", cfg.dxvk_device);
    println!("helper_exe={}", install_dir::helper_exe().map(|p| p.display().to_string()).unwrap_or_else(|| "missing".to_string()));
    for (k, v) in &cfg.settings {
        println!("{k}={v}");
    }
    ExitCode::SUCCESS
}

fn cmd_runners() -> ExitCode {
    let found = neuralforge_supervisor::runners::discover_proton();
    if found.is_empty() {
        if let Some(wine) = neuralforge_supervisor::runners::find_wine() {
            println!("wine\t{}", wine.display());
            return ExitCode::SUCCESS;
        }
        eprintln!("no custom compatibility tools or system wine found");
        return ExitCode::FAILURE;
    }
    for runner in found {
        println!("{}\t{}", runner.name, runner.path.display());
    }
    ExitCode::SUCCESS
}

fn cmd_detect_gpu() -> ExitCode {
    match gpu::detect_nvidia_gpu() {
        Some((vendor, device)) => {
            println!("{vendor:04x}:{device:04x}");
            ExitCode::SUCCESS
        }
        None => {
            eprintln!("no NVIDIA GPU detected");
            ExitCode::FAILURE
        }
    }
}

fn cmd_status() -> ExitCode {
    match neuralforge_supervisor::is_running() {
        Some(pid) => println!("helper running (pid {pid})"),
        None => println!("helper not running"),
    }
    println!("  config: {}", paths::config_file());
    println!("  runtime: {}", neuralforge_protocol::shm_runtime_dir());
    println!("  state: {}", paths::state_dir());
    ExitCode::SUCCESS
}

fn cmd_doctor() -> ExitCode {
    let mut ok = true;
    let cfg = Config::load();

    print!("config: {}\n  ", paths::config_file());
    if std::path::Path::new(&paths::config_file()).exists() {
        println!("ok");
    } else {
        println!("missing (run `neuralforge-cli init`)");
        ok = false;
    }

    let helper = install_dir::helper_exe();
    print!("helper exe: {}\n  ", helper.as_deref().map(|p| p.display().to_string()).unwrap_or_else(|| "missing".to_string()));
    if helper.is_some() {
        println!("ok");
    } else {
        println!("missing");
        ok = false;
    }

    let (runner_type, runner_path) = if !cfg.runner_path.is_empty() {
        (cfg.runner_type.clone(), cfg.runner_path.clone())
    } else if let Some(proton) = neuralforge_supervisor::runners::best_proton() {
        ("proton".to_string(), proton.path.display().to_string())
    } else if let Some(wine) = neuralforge_supervisor::runners::find_wine() {
        ("wine".to_string(), wine.display().to_string())
    } else {
        ("none".to_string(), String::new())
    };
    print!("runner: {runner_type} {runner_path}\n  ");
    if !runner_path.is_empty() && std::path::Path::new(&runner_path).exists() {
        println!("ok");
    } else {
        println!("missing/not executable");
        ok = false;
    }

    let binaries = if cfg.binaries.is_empty() { paths::binaries_dir() } else { cfg.binaries.clone() };
    let ngx_dll = std::path::Path::new(&binaries).join("nvngx_dlssnr.dll");
    print!("binaries: {binaries}\n  nvngx_dlssnr.dll: ");
    if ngx_dll.exists() {
        println!("ok");
    } else {
        println!("error -- missing (required; see `neuralforge-cli import-binaries DIR`)");
        ok = false;
    }

    print!("vendored dxvk dll: {}\n  ", install_dir::dxvk_dll().as_deref().map(|p| p.display().to_string()).unwrap_or_else(|| "missing".to_string()));
    if install_dir::dxvk_dll().is_some() {
        println!("ok");
    } else {
        println!("missing (only needed for the system-Wine fallback runner)");
    }

    print!("runtime dir: {}\n  ", neuralforge_protocol::shm_runtime_dir());
    match paths::ensure_dirs() {
        Ok(()) => println!("ok"),
        Err(e) => {
            println!("not writable: {e}");
            ok = false;
        }
    }

    if ok { ExitCode::SUCCESS } else { ExitCode::FAILURE }
}

fn cmd_setup() -> ExitCode {
    let init_result = cmd_init();
    if init_result != ExitCode::SUCCESS {
        return init_result;
    }
    println!("setup complete");
    ExitCode::SUCCESS
}

fn cmd_start() -> ExitCode {
    let cfg = Config::load();
    match neuralforge_supervisor::start(&cfg) {
        Ok(started) => {
            println!("helper started (pid {})", started.pid);
            println!("  runner: {} {}", started.runner_type, started.runner_path);
            println!("  log: {}", started.log);
            ExitCode::SUCCESS
        }
        Err(neuralforge_supervisor::StartError::AlreadyRunning(_)) => {
            println!("helper already running");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn cmd_stop() -> ExitCode {
    match neuralforge_supervisor::stop(Duration::from_secs(5)) {
        Ok(()) => {
            println!("helper stopped");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("failed to stop helper: {e}");
            ExitCode::FAILURE
        }
    }
}

fn cmd_install(appdir: Option<&String>) -> ExitCode {
    let Some(appdir) = appdir else {
        eprintln!("usage: neuralforge-cli install --appdir DIR");
        return ExitCode::FAILURE;
    };
    match neuralforge_supervisor::install::install(std::path::Path::new(appdir)) {
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

fn cmd_uninstall() -> ExitCode {
    match neuralforge_supervisor::install::uninstall() {
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
        eprintln!("usage: neuralforge-cli import-binaries DIR");
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
    let mut copied = 0;
    for name in ["nvngx_dlssnr.dll", "nvngx.dll", "nvapi64.dll"] {
        let from = src.join(name);
        if from.is_file() {
            if let Err(e) = std::fs::copy(&from, std::path::Path::new(&dest).join(name)) {
                eprintln!("failed to copy {name}: {e}");
                return ExitCode::FAILURE;
            }
            copied += 1;
        }
    }
    println!("imported {copied} file(s) to {dest}");
    ExitCode::SUCCESS
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let Some(command) = args.get(1) else {
        usage();
        return ExitCode::FAILURE;
    };

    match command.as_str() {
        "init" => cmd_init(),
        "setup" => cmd_setup(),
        "start" => cmd_start(),
        "stop" => cmd_stop(),
        "restart" => {
            cmd_stop();
            cmd_start()
        }
        "status" => cmd_status(),
        "doctor" => cmd_doctor(),
        "config" => cmd_config(),
        "runners" => cmd_runners(),
        "detect-gpu" => cmd_detect_gpu(),
        "import-binaries" => cmd_import_binaries(args.get(2)),
        "install" => {
            let appdir = args.iter().position(|a| a == "--appdir").and_then(|i| args.get(i + 1));
            cmd_install(appdir)
        }
        "uninstall" => cmd_uninstall(),
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
