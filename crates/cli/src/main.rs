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
    neural_forge_supervisor::apply_saved_settings(&channel_cfg, &mapping);
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
    neural_forge_supervisor::apply_saved_settings(&channel_cfg, &mapping);
    let header = mapping.header();
    neural_forge_protocol::persist::apply(header, settings);
    // Matches the GUI's own reset-settings flow: applying to the live header alone
    // only affects the running session, so also fold the new values into config.ini
    // via a fresh snapshot (picks up every persisted setting, not just what this
    // profile happened to list) so the change survives a reboot too.
    if let Err(e) = neural_forge_supervisor::save_settings(header) {
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

    print!("user dirs: {}, {}\n  ", paths::data_dir(), paths::state_dir());
    match paths::ensure_dirs() {
        Ok(()) => println!("ok"),
        Err(e) => {
            println!("not writable: {e}");
            ok = false;
        }
    }

    // The channel itself, opened the way the GUI and the layer open it: a runtime dir that exists
    // but is not ours or not private, or a header from another build, fails here.
    print!("channel: {}\n  ", neural_forge_supervisor::channel_path(&cfg));
    match neural_forge_supervisor::open_channel(&cfg) {
        Ok(_) => println!("ok"),
        Err(e) => {
            println!("cannot open: {e}");
            ok = false;
        }
    }

    if ok { ExitCode::SUCCESS } else { ExitCode::FAILURE }
}

fn cmd_install(appdir: &str) -> ExitCode {
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

fn cmd_import_binaries(dir: &str) -> ExitCode {
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
    if !from.is_file() {
        eprintln!("no {name} in {dir}: nothing imported");
        return ExitCode::FAILURE;
    }
    if let Err(e) = std::fs::copy(&from, std::path::Path::new(&dest).join(name)) {
        eprintln!("failed to copy {name}: {e}");
        return ExitCode::FAILURE;
    }
    println!("imported {name} to {dest}");
    ExitCode::SUCCESS
}

fn cmd_extract_model(source: &str) -> ExitCode {
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

/// A command line, parsed and checked before anything runs.
#[derive(Debug, PartialEq, Eq)]
enum Command {
    Init,
    Status,
    Doctor,
    Config,
    ImportBinaries(String),
    ExtractModel(String),
    Install(String),
    Uninstall { purge: bool },
    Shmctl(Vec<String>),
    Profile(Vec<String>),
    /// `--help`/`-h`: the whole usage (`None`) or one command's.
    Help(Option<String>),
}

/// Why a command line was refused: printed with the matching usage, exit code 2.
#[derive(Debug, PartialEq, Eq)]
struct UsageError {
    command: Option<String>,
    message: String,
}

fn is_help(arg: &str) -> bool {
    matches!(arg, "--help" | "-h" | "help")
}

/// Parses `args` (without the program name). Every argument must be recognised: a typo such as
/// `uninstall --purg` is refused rather than ignored, and `<command> --help` only prints help.
fn parse(args: &[String]) -> Result<Command, UsageError> {
    let Some(command) = args.first() else {
        return Err(UsageError { command: None, message: "no command given".into() });
    };
    let rest: Vec<&str> = args[1..].iter().map(String::as_str).collect();
    let err = |message: String| UsageError { command: Some(command.clone()), message };
    if is_help(command) {
        return match rest.as_slice() {
            [] => Ok(Command::Help(None)),
            [topic] if command_usage(topic).is_some() => Ok(Command::Help(Some((*topic).to_string()))),
            _ => Err(UsageError { command: None, message: format!("no help for {:?}", rest.join(" ")) }),
        };
    }
    if command_usage(command).is_none() {
        return Err(UsageError { command: None, message: format!("unknown command: {command}") });
    }
    if rest.first().is_some_and(|a| matches!(*a, "--help" | "-h")) || (rest.first() == Some(&"help") && matches!(command.as_str(), "shmctl" | "profile")) {
        return if rest.len() == 1 { Ok(Command::Help(Some(command.clone()))) } else { Err(err(format!("unexpected argument {:?}", rest[1]))) };
    }
    let unexpected = |extra: &[&str]| err(format!("unexpected argument {:?}", extra[0]));
    let one = |what: &str| match rest.as_slice() {
        [value] if !value.starts_with('-') => Ok((*value).to_string()),
        [] => Err(err(format!("{command} takes {what}"))),
        [value] => Err(err(format!("unknown option {value:?}"))),
        [_, extra @ ..] => Err(unexpected(extra)),
    };
    let none = |cmd: Command| if rest.is_empty() { Ok(cmd) } else { Err(err(format!("unexpected argument {:?}", rest[0]))) };
    match command.as_str() {
        "init" => none(Command::Init),
        "status" => none(Command::Status),
        "doctor" => none(Command::Doctor),
        "config" => none(Command::Config),
        "import-binaries" => one("a directory").map(Command::ImportBinaries),
        "extract-model" => one("a directory or the DLL").map(Command::ExtractModel),
        "install" => match rest.as_slice() {
            ["--appdir", dir] => Ok(Command::Install((*dir).to_string())),
            ["--appdir"] | [] => Err(err("install takes --appdir DIR".into())),
            [first, ..] if *first != "--appdir" => Err(err(format!("unknown option {first:?}"))),
            [_, _, extra @ ..] => Err(unexpected(extra)),
            _ => Err(err("install takes --appdir DIR".into())),
        },
        "uninstall" => match rest.as_slice() {
            [] => Ok(Command::Uninstall { purge: false }),
            ["--purge"] => Ok(Command::Uninstall { purge: true }),
            ["--purge", extra @ ..] => Err(unexpected(extra)),
            [other, ..] => Err(err(format!("unknown option {other:?}"))),
        },
        "shmctl" => shmctl::check_args(&args[1..]).map(|()| Command::Shmctl(args[1..].to_vec())).map_err(err),
        "profile" => {
            let ok = match rest.as_slice() {
                [] | ["list"] => true,
                ["save" | "load" | "delete", name] => !name.starts_with('-'),
                _ => false,
            };
            if ok { Ok(Command::Profile(args[1..].to_vec())) } else { Err(err(format!("unexpected arguments: {}", rest.join(" ")))) }
        }
        _ => unreachable!("command_usage knows every command"),
    }
}

/// One command's usage line, `None` for a command that does not exist.
fn command_usage(command: &str) -> Option<&'static str> {
    Some(match command {
        "init" => "usage: neural-forge-cli init\n  create the default config",
        "status" => "usage: neural-forge-cli status\n  show the config, channel and model",
        "doctor" => "usage: neural-forge-cli doctor\n  check the config, NVIDIA DLL, model and paths",
        "config" => "usage: neural-forge-cli config\n  print the effective config",
        "import-binaries" => "usage: neural-forge-cli import-binaries DIR\n  copy NVIDIA's nvngx_dlssnr.dll from DIR into the user data dir",
        "extract-model" => "usage: neural-forge-cli extract-model DIR\n  write the model directory from nvngx_dlssnr.dll (DIR holds it, or is the DLL)",
        "install" => "usage: neural-forge-cli install --appdir DIR\n  install an extracted AppImage AppDir into persistent user storage",
        "uninstall" => {
            "usage: neural-forge-cli uninstall [--purge]\n  remove unchanged tracked installed files; --purge also removes config, data\n  (the DLL, the model), state and /tmp/neural-forge-$UID"
        }
        "shmctl" => "",
        "profile" => "",
        _ => return None,
    })
}

fn print_help(command: Option<&str>) {
    match command {
        None => usage(),
        Some("shmctl") => shmctl::usage(),
        Some("profile") => profile_usage(),
        Some(command) => eprintln!("{}", command_usage(command).unwrap_or_default()),
    }
}

fn main() -> ExitCode {
    // `args_os`: `args()` panics on an argument that is not valid UTF-8 (a path, say).
    let args: Vec<String> = std::env::args_os().skip(1).map(|a| a.to_string_lossy().into_owned()).collect();
    let command = match parse(&args) {
        Ok(command) => command,
        Err(e) => {
            eprintln!("neural-forge-cli: {}\n", e.message);
            print_help(e.command.as_deref());
            return ExitCode::from(2);
        }
    };
    match command {
        Command::Init => cmd_init(),
        Command::Status => cmd_status(),
        Command::Doctor => cmd_doctor(),
        Command::Config => cmd_config(),
        Command::ImportBinaries(dir) => cmd_import_binaries(&dir),
        Command::ExtractModel(source) => cmd_extract_model(&source),
        Command::Install(appdir) => cmd_install(&appdir),
        Command::Uninstall { purge } => cmd_uninstall(purge),
        Command::Shmctl(args) => shmctl::run(&args),
        Command::Profile(args) => cmd_profile(&args),
        Command::Help(command) => {
            print_help(command.as_deref());
            ExitCode::SUCCESS
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_str(line: &str) -> Result<Command, UsageError> {
        parse(&line.split_whitespace().map(String::from).collect::<Vec<_>>())
    }

    #[test]
    fn uninstall_help_only_prints_help() {
        assert_eq!(parse_str("uninstall --help"), Ok(Command::Help(Some("uninstall".into()))));
        assert_eq!(parse_str("uninstall -h"), Ok(Command::Help(Some("uninstall".into()))));
        assert_eq!(parse_str("uninstall"), Ok(Command::Uninstall { purge: false }));
        assert_eq!(parse_str("uninstall --purge"), Ok(Command::Uninstall { purge: true }));
    }

    #[test]
    fn a_misspelt_or_extra_argument_is_refused() {
        let e = parse_str("uninstall --purg").unwrap_err();
        assert_eq!(e.command.as_deref(), Some("uninstall"));
        assert!(e.message.contains("--purg"), "{}", e.message);
        assert!(parse_str("uninstall --purge now").is_err());
        assert!(parse_str("init now").is_err());
        assert!(parse_str("status --verbose").is_err());
        assert!(parse_str("import-binaries a b").is_err());
        assert!(parse_str("import-binaries").is_err());
        assert!(parse_str("install --appdir a b").is_err());
        assert!(parse_str("install --app a").is_err());
        assert!(parse_str("profile save").is_err());
        assert!(parse_str("profile save a b").is_err());
        assert!(parse_str("shmctl set intensity").is_err());
        assert!(parse_str("shmctl status now").is_err());
        assert!(parse_str("bogus").is_err());
        assert!(parse_str("").is_err());
    }

    #[test]
    fn init_help_and_the_other_help_forms() {
        assert_eq!(parse_str("init --help"), Ok(Command::Help(Some("init".into()))));
        assert_eq!(parse_str("--help"), Ok(Command::Help(None)));
        assert_eq!(parse_str("help install"), Ok(Command::Help(Some("install".into()))));
        assert_eq!(parse_str("shmctl help"), Ok(Command::Help(Some("shmctl".into()))));
        assert_eq!(parse_str("profile --help"), Ok(Command::Help(Some("profile".into()))));
        assert!(parse_str("init --help extra").is_err());
    }

    #[test]
    fn well_formed_commands_parse() {
        assert_eq!(parse_str("init"), Ok(Command::Init));
        assert_eq!(parse_str("install --appdir /x"), Ok(Command::Install("/x".into())));
        assert_eq!(parse_str("extract-model /d"), Ok(Command::ExtractModel("/d".into())));
        assert_eq!(parse_str("profile load gta"), Ok(Command::Profile(vec!["load".into(), "gta".into()])));
        assert_eq!(parse_str("profile"), Ok(Command::Profile(vec![])));
        assert_eq!(parse_str("shmctl set intensity 0.5"), Ok(Command::Shmctl(vec!["set".into(), "intensity".into(), "0.5".into()])));
        assert_eq!(parse_str("shmctl capture --frames 3"), Ok(Command::Shmctl(vec!["capture".into(), "--frames".into(), "3".into()])));
    }
}
