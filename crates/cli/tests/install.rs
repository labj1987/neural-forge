//! `install` and `uninstall` run as a person runs them: the CLI binary against an AppDir, every XDG home (and HOME)
//! in a scratch directory, so the real ones are never seen.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const APP_ID: &str = "io.github.labj1987.NeuralForge";

/// A scratch directory, removed when the test ends, passed or failed.
struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn scratch(tag: &str) -> Scratch {
    let dir = std::env::temp_dir().join(format!("neural-forge-install-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    Scratch(dir)
}

/// The runtime dir's uid: unique, so it names a directory no real session owns.
fn uid(tag: &str) -> String {
    format!("installtest-{tag}-{}", std::process::id())
}

fn cli(root: &Path, tag: &str, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_neural-forge-cli"))
        .args(args)
        // HOME too: purge looks for Steam (and its launch options) under it.
        .env("HOME", root.join("home"))
        .env("XDG_DATA_HOME", root.join("data"))
        .env("XDG_CONFIG_HOME", root.join("config"))
        .env("XDG_STATE_HOME", root.join("state"))
        .env("NEURAL_FORGE_UID", uid(tag))
        .output()
        .expect("run neural-forge-cli")
}

fn ok(out: &Output) -> bool {
    out.status.success()
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// The AppDir every test installs: what build-appimage.sh puts in it, in miniature.
fn write_appdir(app: &Path) {
    let manifest = r#"{"layer": {"name": "VK_LAYER_neuralforge_neural"}}"#;
    let desktop = format!("share/applications/{APP_ID}.desktop");
    let appdata = format!("share/metainfo/{APP_ID}.appdata.xml");
    let files = [
        ("bin/neural-forge", "gui"),
        ("bin/neural-forge-cli", "cli"),
        ("lib/neural-forge/libneural_forge_layer.so", "layer"),
        (desktop.as_str(), "[Desktop Entry]\nExec=neural-forge\n"),
        ("share/icons/hicolor/scalable/apps/neural-forge.svg", "<svg/>"),
        (appdata.as_str(), "<component/>"),
        ("share/vulkan/implicit_layer.d/neural_forge_layer.json", manifest),
    ];
    for (name, text) in files {
        let path = app.join("usr").join(name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }
}

/// Another program's layer manifest, which nothing here may touch.
fn write_upstream(data: &Path) -> PathBuf {
    let upstream = data.join("vulkan/implicit_layer.d/VkLayer_DLSS5.json");
    std::fs::create_dir_all(upstream.parent().unwrap()).unwrap();
    std::fs::write(&upstream, "upstream sentinel").unwrap();
    upstream
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap()
}

/// What the CLI checks before it edits Steam's launch options: a `steam` or `steamwebhelper` process.
fn steam_running() -> bool {
    let Ok(entries) = std::fs::read_dir("/proc") else { return false };
    entries
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().bytes().all(|b| b.is_ascii_digit()))
        .any(|e| std::fs::read_to_string(e.path().join("comm")).is_ok_and(|comm| matches!(comm.trim_end_matches('\n'), "steam" | "steamwebhelper")))
}

#[test]
fn coexistence_and_owned_removal() {
    let dir = scratch("owned");
    let root = &dir.0;
    let data = root.join("data");
    let app = root.join("AppDir");
    write_appdir(&app);
    let upstream = write_upstream(&data);
    let appdir = app.to_str().unwrap();
    let install = || cli(root, "owned", &["install", "--appdir", appdir]);

    for _ in 0..2 {
        let out = install();
        assert!(ok(&out), "{}", stderr(&out));
    }
    // An update replaces the file by rename: a program running from the old one keeps reading the old image.
    let binary = data.join("neural-forge/bin/neural-forge");
    let mut running_image = std::fs::File::open(&binary).unwrap();
    std::fs::write(app.join("usr/bin/neural-forge"), "updated gui").unwrap();
    let out = install();
    assert!(ok(&out), "{}", stderr(&out));
    let mut old = String::new();
    running_image.read_to_string(&mut old).unwrap();
    assert_eq!(old, "gui");
    assert_eq!(read(&binary), "updated gui");

    // A file the user changed is neither overwritten nor removed.
    std::fs::write(&binary, "user changed").unwrap();
    let out = install();
    assert!(!ok(&out), "an install over a changed file must fail");
    let out = cli(root, "owned", &["uninstall"]);
    assert!(ok(&out), "{}", stderr(&out));
    assert_eq!(read(&binary), "user changed");
    assert_eq!(read(&upstream), "upstream sentinel");
    assert!(!data.join("neural-forge/bin/neural-forge-cli").exists());
}

#[test]
fn purge_removes_everything_ours_and_nothing_else() {
    let dir = scratch("purge");
    let root = &dir.0;
    let data = root.join("data");
    let app = root.join("AppDir");
    write_appdir(&app);
    let upstream = write_upstream(&data);
    // Outside the scratch dir: removed when the test ends whichever way it ends (with Steam running the purge is
    // refused and leaves it).
    let runtime = Scratch(PathBuf::from(format!("/tmp/neural-forge-{}", uid("purge"))));

    let out = cli(root, "purge", &["install", "--appdir", app.to_str().unwrap()]);
    assert!(ok(&out), "{}", stderr(&out));
    for (path, text) in [
        ("data/neural-forge/binaries/nvngx_dlssnr.dll", "dll"),
        ("config/neural-forge/config.ini", "set_intensity=1\n"),
        ("state/neural-forge/state.txt", "state"),
    ] {
        let path = root.join(path);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }
    std::fs::create_dir_all(&runtime.0).unwrap();
    std::fs::write(runtime.0.join("shm.bin"), "shm").unwrap();
    // What the install really put in place (and nothing for menu integration: that is the AppImage integrator's job).
    let installed: Vec<PathBuf> = [
        "neural-forge/bin/neural-forge",
        "neural-forge/bin/neural-forge-cli",
        "neural-forge/lib/neural-forge/libneural_forge_layer.so",
        "vulkan/implicit_layer.d/neural_forge_layer.json",
        "neural-forge/installation.json",
    ]
    .iter()
    .map(|p| data.join(p))
    .collect();
    for path in &installed {
        assert!(path.is_file(), "{}", path.display());
    }
    assert!(!data.join("applications").exists(), "the installer writes no menu entry");

    // A Steam account with a game switched on, and the copies of its localconfig.vdf an edit left.
    let steam_config = root.join("home/.local/share/Steam/userdata/1/config");
    std::fs::create_dir_all(root.join("home/.local/share/Steam/steamapps")).unwrap();
    std::fs::create_dir_all(&steam_config).unwrap();
    let localconfig = steam_config.join("localconfig.vdf");
    let text = "\"UserLocalConfigStore\"\n{\n\t\"Software\"\n\t{\n\t\t\"Valve\"\n\t\t{\n\t\t\t\"Steam\"\n\t\t\t{\n\t\t\t\t\"apps\"\n\
                \t\t\t\t{\n\t\t\t\t\t\"3240220\"\n\t\t\t\t\t{\n\t\t\t\t\t\t\"LaunchOptions\"\t\t\"NEURAL_FORGE_ENABLE=1 \
                NEURAL_FORGE_TARGET_EXE=G.exe gamemoderun %command%\"\n\t\t\t\t\t}\n\t\t\t\t}\n\t\t\t}\n\t\t}\n\t}\n}\n";
    std::fs::write(&localconfig, text).unwrap();
    let copies: Vec<PathBuf> = ["orig", "bak", "lock"].iter().map(|s| steam_config.join(format!("localconfig.vdf.neural-forge.{s}"))).collect();
    for copy in &copies {
        std::fs::write(copy, text).unwrap();
    }
    // A lost record: purge still takes the manifest that would load the deleted library.
    std::fs::remove_file(data.join("neural-forge/installation.json")).unwrap();

    if steam_running() {
        // This machine's Steam is running (the purge reads the real /proc): refused, nothing removed.
        let out = cli(root, "purge", &["uninstall", "--purge"]);
        assert!(!ok(&out));
        assert!(stderr(&out).contains("app 3240220: Steam is running"), "{}", stderr(&out));
        assert!(stderr(&out).contains("nothing was removed"), "{}", stderr(&out));
        assert_eq!(read(&localconfig), text);
        for kept in installed[..installed.len() - 1].iter().chain(&copies) {
            assert!(kept.exists(), "{}", kept.display());
        }
        return;
    }
    let out = cli(root, "purge", &["uninstall", "--purge"]);
    assert!(ok(&out), "{}", stderr(&out));
    let ours = [data.join("neural-forge"), root.join("config/neural-forge"), root.join("state/neural-forge"), runtime.0.clone()];
    for gone in installed.iter().chain(&copies).chain(&ours) {
        assert!(!gone.exists(), "{}", gone.display());
    }
    assert!(!read(&localconfig).contains("NEURAL_FORGE"));
    assert!(read(&localconfig).contains("\"gamemoderun %command%\""));
    assert_eq!(read(&upstream), "upstream sentinel");
}
