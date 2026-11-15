//! The desktop's own HDR state, for the HDR verdict (the layer's `hdr_verdict` says what the game asked for; this
//! says whether the screen it goes to is in HDR). Read without root and without changing anything, from the
//! compositor the session runs ([`Desktop::from_env`]):
//!
//! - GNOME: `org.gnome.Mutter.DisplayConfig.GetCurrentState` over the session bus (`gdbus`; each monitor's
//!   `color-mode`, 1 being BT.2100);
//! - KDE Plasma: `kscreen-doctor -j` (each output's `hdr`);
//! - Hyprland: `hyprctl monitors -j` (each monitor's `colorManagementPreset`: `hdr` or `hdredid` is HDR; any other
//!   preset is not HDR itself, but with `render:cm_auto_hdr`, on by default, a fullscreen HDR game switches the display
//!   to HDR while the preset still reads the same, so that is said rather than "off");
//! - sway: `swaymsg -t get_outputs -r` (each output's `hdr`);
//! - a gamescope session, from its environment.
//!
//! Hyprland and sway are told by their own variables only while the socket they name exists: a stale one left in the
//! systemd user manager's environment after the compositor ended must not override the desktop that runs now.
//!
//! COSMIC: cosmic-comp has no HDR output yet and nothing reports one, so it says so. When it gets one, its
//! reader is one more [`Probe`] in [`Desktop::probe`].
//!
//! Where none of them answers the text says the state was not detected and why; it never guesses "off".

use crate::doctor::{run_command, CommandOutput};

const TIMEOUT: std::time::Duration = std::time::Duration::from_secs(3);

/// One display as the compositor describes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Output {
    pub name: String,
    /// HDR on or off; `None` where the compositor's answer does not say (a Hyprland preset that is not HDR itself).
    pub hdr: Option<bool>,
    /// Said after the name instead of "HDR on/off" when `hdr` is `None`, and in brackets after "HDR on" otherwise
    /// (Hyprland's preset); empty: nothing.
    pub detail: String,
}

/// The compositors whose HDR state can be asked for (or, for COSMIC, is known not to be reported).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Desktop {
    Gamescope,
    Gnome,
    Kde,
    Hyprland,
    Sway,
    Cosmic,
}

/// How a desktop's HDR state is read: the command, and the parser for what it prints.
struct Probe {
    program: &'static str,
    args: &'static [&'static str],
    parse: fn(&str) -> Vec<Output>,
    /// Said after "HDR on" (GNOME's mode is named BT.2100).
    on_detail: &'static str,
    /// Said after the list when a display's state is not known as on or off (what can still turn HDR on there).
    unknown_note: &'static str,
}

impl Desktop {
    /// The name the text gives it.
    fn name(self) -> &'static str {
        match self {
            Desktop::Gamescope => "gamescope session",
            Desktop::Gnome => "GNOME",
            Desktop::Kde => "KDE",
            Desktop::Hyprland => "Hyprland",
            Desktop::Sway => "sway",
            Desktop::Cosmic => "COSMIC",
        }
    }

    /// The session's compositor from its environment (`var` reads a variable, empty counting as unset; `exists`
    /// says whether a path exists): a gamescope session (`GAMESCOPE_WAYLAND_DISPLAY`), Hyprland
    /// (`HYPRLAND_INSTANCE_SIGNATURE`, while `$XDG_RUNTIME_DIR/hypr/<signature>/.socket.sock` exists) and sway (`SWAYSOCK`, while that socket exists) by the variables
    /// they set for their clients, which are there even when a session started from a console has no
    /// `XDG_CURRENT_DESKTOP` (and a nested Hyprland keeps its parent's); then `XDG_CURRENT_DESKTOP`'s
    /// colon-separated names (`ubuntu:GNOME`, `KDE`, `Hyprland`, `sway:wlroots`, `COSMIC`), the first one known.
    fn from_env(var: &dyn Fn(&str) -> Option<String>, exists: &dyn Fn(&std::path::Path) -> bool) -> Option<Desktop> {
        let get = |name: &str| var(name).filter(|v| !v.is_empty());
        if get("GAMESCOPE_WAYLAND_DISPLAY").is_some() {
            return Some(Desktop::Gamescope);
        }
        if let Some(signature) = get("HYPRLAND_INSTANCE_SIGNATURE") {
            let socket = get("XDG_RUNTIME_DIR").map(|dir| std::path::Path::new(&dir).join("hypr").join(&signature).join(".socket.sock"));
            if socket.is_some_and(|socket| exists(&socket)) {
                return Some(Desktop::Hyprland);
            }
        }
        if get("SWAYSOCK").is_some_and(|socket| exists(std::path::Path::new(&socket))) {
            return Some(Desktop::Sway);
        }
        let current = var("XDG_CURRENT_DESKTOP").unwrap_or_default();
        current.split(':').find_map(|name| match name.trim().to_ascii_lowercase().as_str() {
            "gnome" => Some(Desktop::Gnome),
            "kde" => Some(Desktop::Kde),
            "hyprland" => Some(Desktop::Hyprland),
            "sway" => Some(Desktop::Sway),
            "cosmic" => Some(Desktop::Cosmic),
            _ => None,
        })
    }

    /// How its HDR state is read; `Err` says why it cannot be.
    fn probe(self) -> Result<Probe, &'static str> {
        match self {
            Desktop::Gamescope => Err("HDR is gamescope's (--hdr-enabled) and the display's"),
            Desktop::Gnome => Ok(Probe {
                program: "gdbus",
                args: &[
                    "call",
                    "--session",
                    "--dest",
                    "org.gnome.Mutter.DisplayConfig",
                    "--object-path",
                    "/org/gnome/Mutter/DisplayConfig",
                    "--method",
                    "org.gnome.Mutter.DisplayConfig.GetCurrentState",
                ],
                parse: parse_mutter,
                on_detail: " (BT.2100)",
                unknown_note: "",
            }),
            Desktop::Kde => Ok(Probe { program: "kscreen-doctor", args: &["-j"], parse: parse_kscreen, on_detail: "", unknown_note: "" }),
            // A monitor's preset is what its rule asks for: with `render:cm_auto_hdr` (on by default) Hyprland's renderer
            // switches a monitor to HDR for a fullscreen HDR client while the preset reads the same.
            Desktop::Hyprland => Ok(Probe {
                program: "hyprctl",
                args: &["monitors", "-j"],
                parse: parse_hyprland,
                on_detail: "",
                unknown_note: "; with render:cm_auto_hdr, on by default, a fullscreen HDR game switches the display to HDR",
            }),
            Desktop::Sway => Ok(Probe { program: "swaymsg", args: &["-t", "get_outputs", "-r"], parse: parse_sway, on_detail: "", unknown_note: "" }),
            Desktop::Cosmic => Err("HDR state not detected (COSMIC does not report HDR yet: cosmic-comp has no HDR output)"),
        }
    }
}

/// The desktop's HDR state per display: "GNOME: HDMI-1 HDR on (BT.2100)", "KDE: DP-1 HDR on, HDMI-A-1 HDR off",
/// "Hyprland: DP-1 preset srgb (not HDR itself); with render:cm_auto_hdr, ...", or why it is not known ("sway: HDR state not detected (swaymsg is not installed)",
/// "HDR state not detected on this desktop (XDG_CURRENT_DESKTOP=X-Cinnamon)").
pub fn read() -> String {
    read_with(&|name| std::env::var(name).ok(), &|path| path.exists(), &|program, args| run_command(program, args, TIMEOUT))
}

/// [`read`] with the environment, the file system's answer to "does this path exist" and the commands given.
fn read_with(var: &dyn Fn(&str) -> Option<String>, exists: &dyn Fn(&std::path::Path) -> bool, run: &dyn Fn(&str, &[&str]) -> CommandOutput) -> String {
    match Desktop::from_env(var, exists) {
        Some(desktop) => match desktop.probe() {
            Ok(probe) => match ask(&probe, run) {
                Ok(outputs) => describe(desktop.name(), &outputs, &probe),
                Err(why) => format!("{}: HDR state not detected ({why})", desktop.name()),
            },
            Err(why) => format!("{}: {why}", desktop.name()),
        },
        // No desktop named: GNOME and KDE answer over their own interfaces, so ask them.
        None => {
            for desktop in [Desktop::Gnome, Desktop::Kde] {
                if let Ok(probe) = desktop.probe() {
                    if let Ok(outputs) = ask(&probe, run) {
                        return describe(desktop.name(), &outputs, &probe);
                    }
                }
            }
            match var("XDG_CURRENT_DESKTOP").filter(|v| !v.is_empty()) {
                Some(named) => format!("HDR state not detected on this desktop (XDG_CURRENT_DESKTOP={named})"),
                None => "HDR state not detected on this desktop (it names none: XDG_CURRENT_DESKTOP is not set)".into(),
            }
        }
    }
}

/// Runs `probe`: the displays it lists, or why there are none.
fn ask(probe: &Probe, run: &dyn Fn(&str, &[&str]) -> CommandOutput) -> Result<Vec<Output>, String> {
    match run(probe.program, probe.args) {
        CommandOutput::Ran(text) => {
            let outputs = (probe.parse)(&text);
            if outputs.is_empty() {
                Err(format!("{} reported no HDR state for any display", probe.program))
            } else {
                Ok(outputs)
            }
        }
        CommandOutput::Failed(why) => Err(format!("{} failed: {why}", probe.program)),
        CommandOutput::Missing => Err(format!("{} is not installed", probe.program)),
    }
}

fn describe(desktop: &str, outputs: &[Output], probe: &Probe) -> String {
    let list: Vec<String> = outputs
        .iter()
        .map(|o| {
            let detail = if o.detail.is_empty() { String::new() } else { format!(" ({})", o.detail) };
            match o.hdr {
                Some(true) => format!("{} HDR on{}{detail}", o.name, probe.on_detail),
                Some(false) => format!("{} HDR off", o.name),
                None => format!("{} {}", o.name, o.detail),
            }
        })
        .collect();
    let note = if outputs.iter().any(|o| o.hdr.is_none()) { probe.unknown_note } else { "" };
    format!("{desktop}: {}{note}", list.join(", "))
}

/// The monitors in Mutter's `GetCurrentState` reply (as `gdbus` prints it): each monitor's connector and its
/// `color-mode` (0 default, 1 BT.2100). A monitor spec starts `(('CONNECTOR', 'vendor', ...` and its properties,
/// with `'color-mode': <uint32 N>`, come before the next one.
pub fn parse_mutter(text: &str) -> Vec<Output> {
    let mut out = Vec::new();
    let starts: Vec<usize> = text.match_indices("(('").map(|(i, _)| i).collect();
    for (k, &start) in starts.iter().enumerate() {
        let end = starts.get(k + 1).copied().unwrap_or(text.len());
        let part = &text[start..end];
        let Some(name) = part[3..].split('\'').next().filter(|n| !n.is_empty()) else { continue };
        let Some(at) = part.find("'color-mode': <uint32 ") else { continue };
        let mode = part[at + 22..].split('>').next().and_then(|v| v.trim().parse::<u32>().ok());
        if let Some(mode) = mode {
            out.push(Output { name: name.to_string(), hdr: Some(mode == 1), detail: String::new() });
        }
    }
    out
}

/// The enabled outputs in `kscreen-doctor -j`'s JSON (`outputs[]`), each with its `hdr` flag (absent: off).
pub fn parse_kscreen(text: &str) -> Vec<Output> {
    let Ok(json) = serde_json::from_str::<serde_json::Value>(text) else { return Vec::new() };
    let Some(outputs) = json.get("outputs").and_then(|o| o.as_array()) else { return Vec::new() };
    outputs
        .iter()
        .filter(|o| o.get("enabled").and_then(|e| e.as_bool()).unwrap_or(true))
        .filter_map(|o| {
            Some(Output {
                name: o.get("name")?.as_str()?.to_string(),
                hdr: Some(o.get("hdr").and_then(|h| h.as_bool()).unwrap_or(false)),
                detail: String::new(),
            })
        })
        .collect()
}

/// The monitors in `hyprctl monitors -j`'s JSON array (it lists only enabled ones; `disabled` is skipped anyway),
/// each with its colour-management preset `colorManagementPreset` (Hyprland 0.52 and newer; `NCMType::toString`:
/// `auto`, `srgb`, `wide`, `edid`, `hdr`, `hdredid`, `dcip3`, `dp3`, `adobe`): `hdr` and `hdredid` (PQ with BT.2020
/// primaries) are HDR output. Any other preset is not HDR itself, but with `render:cm_auto_hdr` (on by default) a
/// fullscreen HDR game switches the monitor to HDR while the preset reads the same, so its state is not known: the
/// preset is said as what it is. Without the field (Hyprland before 0.52, which already had HDR
/// presets and `render:cm_auto_hdr` but did not report them) the state is not known either.
pub fn parse_hyprland(text: &str) -> Vec<Output> {
    let Ok(serde_json::Value::Array(monitors)) = serde_json::from_str::<serde_json::Value>(text) else { return Vec::new() };
    monitors
        .iter()
        .filter(|m| !m.get("disabled").and_then(|d| d.as_bool()).unwrap_or(false))
        .filter_map(|m| {
            let name = m.get("name")?.as_str()?.to_string();
            Some(match m.get("colorManagementPreset").and_then(|p| p.as_str()) {
                None => Output { name, hdr: None, detail: "HDR state not reported (Hyprland before 0.52)".into() },
                Some(preset @ ("hdr" | "hdredid")) => Output { name, hdr: Some(true), detail: format!("cm {preset}") },
                Some(preset) => Output { name, hdr: None, detail: format!("preset {preset} (not HDR itself)") },
            })
        })
        .collect()
}

/// The active outputs in `swaymsg -t get_outputs -r`'s JSON array, each with its `hdr` flag (whether HDR is on;
/// sway 1.12 and newer. Absent, as before sway had HDR output: off).
pub fn parse_sway(text: &str) -> Vec<Output> {
    let Ok(serde_json::Value::Array(outputs)) = serde_json::from_str::<serde_json::Value>(text) else { return Vec::new() };
    outputs
        .iter()
        .filter(|o| o.get("active").and_then(|a| a.as_bool()).unwrap_or(true))
        .filter_map(|o| {
            Some(Output {
                name: o.get("name")?.as_str()?.to_string(),
                hdr: Some(o.get("hdr").and_then(|h| h.as_bool()).unwrap_or(false)),
                detail: String::new(),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn out(name: &str, hdr: bool) -> Output {
        Output { name: name.into(), hdr: Some(hdr), detail: String::new() }
    }

    /// Every path exists: the compositors' sockets are there.
    fn live(_: &std::path::Path) -> bool {
        true
    }

    fn env(vars: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
        move |name| vars.iter().find(|(k, _)| *k == name).map(|(_, v)| v.to_string())
    }

    #[test]
    fn mutter_monitors_are_read_with_their_colour_mode() {
        // Shortened from GNOME 49's reply on the test machine.
        let text = "(uint32 1, [(('HDMI-1', 'TCL', 'Beyond TV', '0x00010000'), [('4096x2160@59.940', 4096, 2160, 59.94, 1.0, [1.0], @a{sv} {})], \
                    {'is-builtin': <false>, 'display-name': <'Technical Concepts Ltd 55\"'>, 'color-mode': <uint32 1>, 'supported-color-modes': <[uint32 0, 2, 1]>}), \
                    (('DP-2', 'DEL', 'U2720Q', 'X'), [('3840x2160@60.000', 3840, 2160, 60.0, 1.0, [1.0], {})], {'color-mode': <uint32 0>})], \
                    [(0, 0, 1.0, uint32 0, true, [('HDMI-1', 'TCL', 'Beyond TV', '0x00010000')], @a{sv} {})], {'layout-mode': <uint32 1>})";
        let outputs = parse_mutter(text);
        assert_eq!(outputs, vec![out("HDMI-1", true), out("DP-2", false)]);
        assert_eq!(describe("GNOME", &outputs, &Desktop::Gnome.probe().unwrap()), "GNOME: HDMI-1 HDR on (BT.2100), DP-2 HDR off");
        assert!(parse_mutter("Error: GDBus.Error:org.freedesktop.DBus.Error.ServiceUnknown").is_empty());
    }

    #[test]
    fn kscreen_outputs_are_read_with_their_hdr_flag() {
        let text = r#"{"outputs": [{"name": "DP-1", "enabled": true, "hdr": true}, {"name": "HDMI-A-1", "enabled": true}, {"name": "eDP-1", "enabled": false, "hdr": true}]}"#;
        assert_eq!(parse_kscreen(text), vec![out("DP-1", true), out("HDMI-A-1", false)]);
        assert!(parse_kscreen("not json").is_empty());
    }

    #[test]
    fn hyprland_monitors_are_read_with_their_colour_management_preset() {
        // `hyprctl monitors -j` as Hyprland's own format string prints it (getMonitorData, src/debug/HyprCtl.cpp at
        // v0.52.0; src/ipc/s1/Commands.cpp since v0.57.0, which adds `hardwareDetails`), shortened, with the values of
        // the hyprland-wiki's `hyprctl monitors` example (content/configuring/core/monitors/_index.md). No captured
        // JSON reply is published; the preset strings are `NCMType::toString`'s (src/helpers/CMType.cpp).
        let text = r#"[{
    "id": 0, "name": "DP-1", "description": "Dell Inc. DELL U2723QE", "make": "Dell Inc.", "model": "DELL U2723QE",
    "width": 2560, "height": 1440, "refreshRate": 59.95100, "x": 0, "y": 0, "disabled": false,
    "currentFormat": "XRGB2101010", "mirrorOf": "none", "availableModes": ["2560x1440@59.95Hz"],
    "colorManagementPreset": "hdr", "sdrBrightness": 1.00, "sdrSaturation": 1.00, "sdrMinLuminance": 0.20, "sdrMaxLuminance": 80,
    "hardwareCursorsInUse": true,
    "hardwareDetails": {"backend": "drm", "hdr": true, "chroma": true, "bt2020": true, "vrrCapable": true}
},{
    "id": 1, "name": "HDMI-A-1", "make": "LG Electronics", "model": "LG TV", "width": 3840, "height": 2160, "disabled": false,
    "currentFormat": "XRGB8888", "colorManagementPreset": "srgb", "sdrBrightness": 1.00, "sdrSaturation": 1.00
},{
    "id": 2, "name": "eDP-1", "disabled": true, "colorManagementPreset": "hdredid"
}]"#;
        let outputs = parse_hyprland(text);
        assert_eq!(
            outputs,
            vec![
                Output { name: "DP-1".into(), hdr: Some(true), detail: "cm hdr".into() },
                Output { name: "HDMI-A-1".into(), hdr: None, detail: "preset srgb (not HDR itself)".into() }
            ]
        );
        assert_eq!(
            describe("Hyprland", &outputs, &Desktop::Hyprland.probe().unwrap()),
            "Hyprland: DP-1 HDR on (cm hdr), HDMI-A-1 preset srgb (not HDR itself); with render:cm_auto_hdr, on by default, a fullscreen HDR game \
             switches the display to HDR"
        );
        // `hdredid` is HDR too; `auto` (the default) is never said to be off; a Hyprland before 0.52 reports no preset at all.
        assert_eq!(parse_hyprland(r#"[{"name": "DP-2", "colorManagementPreset": "hdredid"}]"#)[0].hdr, Some(true));
        let auto = parse_hyprland(r#"[{"name": "DP-2", "colorManagementPreset": "auto"}]"#);
        assert_eq!(auto[0].hdr, None);
        assert!(!describe("Hyprland", &auto, &Desktop::Hyprland.probe().unwrap()).contains("HDR off"));
        assert_eq!(
            parse_hyprland(r#"[{"name": "DP-2", "currentFormat": "XRGB8888"}]"#),
            vec![Output { name: "DP-2".into(), hdr: None, detail: "HDR state not reported (Hyprland before 0.52)".into() }]
        );
        assert!(parse_hyprland("HYPRLAND_INSTANCE_SIGNATURE was not set! (Is Hyprland running?)").is_empty());
    }

    #[test]
    fn sway_outputs_are_read_with_their_hdr_flag() {
        // The GET_OUTPUTS example in sway's sway-ipc(7) (sway/sway-ipc.7.scd), shortened, with the `hdr` and
        // `features` members sway 1.12 adds to every output (sway/ipc-json.c), and a second, inactive output.
        let text = r#"[
  {
    "name": "HDMI-A-2", "make": "Unknown", "model": "NS-19E310A13", "serial": "0x00000001",
    "active": true, "primary": false, "scale": 1.0, "subpixel_hinting": "rgb", "transform": "normal",
    "current_workspace": "1", "modes": [{"width": 1920, "height": 1080, "refresh": 60000}],
    "current_mode": {"width": 1920, "height": 1080, "refresh": 60000},
    "features": {"adaptive_sync": true, "hdr": true}, "hdr": true
  },
  {"name": "DP-3", "active": true, "features": {"adaptive_sync": false, "hdr": false}, "hdr": false},
  {"name": "DP-4", "active": false, "hdr": true}
]"#;
        assert_eq!(parse_sway(text), vec![out("HDMI-A-2", true), out("DP-3", false)]);
        // sway before 1.12 has no `hdr`: off.
        assert_eq!(parse_sway(r#"[{"name": "HDMI-A-2", "active": true}]"#), vec![out("HDMI-A-2", false)]);
        assert!(parse_sway("not json").is_empty());
    }

    #[test]
    fn the_desktop_is_told_by_its_environment() {
        assert_eq!(Desktop::from_env(&env(&[("XDG_CURRENT_DESKTOP", "ubuntu:GNOME")]), &live), Some(Desktop::Gnome));
        assert_eq!(Desktop::from_env(&env(&[("XDG_CURRENT_DESKTOP", "KDE")]), &live), Some(Desktop::Kde));
        // sway's session file names `sway:wlroots`; Hyprland sets `Hyprland`; cosmic-session sets `COSMIC`.
        assert_eq!(Desktop::from_env(&env(&[("XDG_CURRENT_DESKTOP", "sway:wlroots")]), &live), Some(Desktop::Sway));
        assert_eq!(Desktop::from_env(&env(&[("XDG_CURRENT_DESKTOP", "Hyprland")]), &live), Some(Desktop::Hyprland));
        assert_eq!(Desktop::from_env(&env(&[("XDG_CURRENT_DESKTOP", "COSMIC")]), &live), Some(Desktop::Cosmic));
        // A session started from a console names none, but the compositor's own variable is there.
        assert_eq!(
            Desktop::from_env(&env(&[("HYPRLAND_INSTANCE_SIGNATURE", "abc_123_456"), ("XDG_RUNTIME_DIR", "/run/user/1000")]), &live),
            Some(Desktop::Hyprland)
        );
        assert_eq!(Desktop::from_env(&env(&[("SWAYSOCK", "/run/user/1000/sway-ipc.1000.42.sock")]), &live), Some(Desktop::Sway));
        assert_eq!(Desktop::from_env(&env(&[("GAMESCOPE_WAYLAND_DISPLAY", "gamescope-0"), ("XDG_CURRENT_DESKTOP", "KDE")]), &live), Some(Desktop::Gamescope));
        assert_eq!(Desktop::from_env(&env(&[("XDG_CURRENT_DESKTOP", "X-Cinnamon"), ("SWAYSOCK", "")]), &live), None);
        assert_eq!(Desktop::from_env(&env(&[]), &live), None);
    }

    #[test]
    fn a_stale_compositor_variable_does_not_override_the_desktop_that_runs() {
        let sockets =
            |path: &std::path::Path| path == std::path::Path::new("/run/user/1000/hypr/abc_123/.socket.sock") || path == std::path::Path::new("/run/s.sock");
        let gone = |_: &std::path::Path| false;
        // A nested Hyprland keeps its parent's XDG_CURRENT_DESKTOP, so the live socket wins over it; a stale variable
        // (left in the systemd user manager's environment) does not. Hyprland's socket is under $XDG_RUNTIME_DIR/hypr/<signature>/; sway's is $SWAYSOCK itself.
        let hypr = env(&[("HYPRLAND_INSTANCE_SIGNATURE", "abc_123"), ("XDG_RUNTIME_DIR", "/run/user/1000"), ("XDG_CURRENT_DESKTOP", "GNOME")]);
        assert_eq!(Desktop::from_env(&hypr, &sockets), Some(Desktop::Hyprland));
        assert_eq!(Desktop::from_env(&hypr, &gone), Some(Desktop::Gnome));
        let sway = env(&[("SWAYSOCK", "/run/s.sock"), ("XDG_CURRENT_DESKTOP", "KDE")]);
        assert_eq!(Desktop::from_env(&sway, &sockets), Some(Desktop::Sway));
        assert_eq!(Desktop::from_env(&sway, &gone), Some(Desktop::Kde));
    }

    #[test]
    fn an_unknown_state_is_said_never_guessed_off() {
        let missing = |_: &str, _: &[&str]| CommandOutput::Missing;
        assert_eq!(read_with(&env(&[("SWAYSOCK", "/run/s.sock")]), &live, &missing), "sway: HDR state not detected (swaymsg is not installed)");
        assert_eq!(
            read_with(&env(&[("XDG_CURRENT_DESKTOP", "Hyprland")]), &live, &|_, _| CommandOutput::Failed("did not finish within 3 s and was stopped".into())),
            "Hyprland: HDR state not detected (hyprctl failed: did not finish within 3 s and was stopped)"
        );
        assert_eq!(
            read_with(&env(&[("XDG_CURRENT_DESKTOP", "KDE")]), &live, &|_, _| CommandOutput::Ran("{}".into())),
            "KDE: HDR state not detected (kscreen-doctor reported no HDR state for any display)"
        );
        assert_eq!(
            read_with(&env(&[("XDG_CURRENT_DESKTOP", "COSMIC")]), &live, &missing),
            "COSMIC: HDR state not detected (COSMIC does not report HDR yet: cosmic-comp has no HDR output)"
        );
        assert_eq!(
            read_with(&env(&[("XDG_CURRENT_DESKTOP", "X-Cinnamon")]), &live, &missing),
            "HDR state not detected on this desktop (XDG_CURRENT_DESKTOP=X-Cinnamon)"
        );
        assert_eq!(read_with(&env(&[]), &live, &missing), "HDR state not detected on this desktop (it names none: XDG_CURRENT_DESKTOP is not set)");
        assert_eq!(
            read_with(&env(&[("GAMESCOPE_WAYLAND_DISPLAY", "gamescope-0")]), &live, &missing),
            "gamescope session: HDR is gamescope's (--hdr-enabled) and the display's"
        );
    }

    #[test]
    fn each_desktop_asks_its_own_tool_and_an_unnamed_one_tries_gnome_then_kde() {
        let asked = std::cell::RefCell::new(Vec::new());
        let run = |program: &str, _: &[&str]| {
            asked.borrow_mut().push(program.to_string());
            match program {
                "swaymsg" => CommandOutput::Ran(r#"[{"name": "DP-1", "active": true, "hdr": true}]"#.into()),
                "kscreen-doctor" => CommandOutput::Ran(r#"{"outputs": [{"name": "DP-1", "hdr": false}]}"#.into()),
                _ => CommandOutput::Missing,
            }
        };
        assert_eq!(read_with(&env(&[("XDG_CURRENT_DESKTOP", "sway:wlroots")]), &live, &run), "sway: DP-1 HDR on");
        assert_eq!(*asked.borrow(), ["swaymsg"]);
        asked.borrow_mut().clear();
        assert_eq!(read_with(&env(&[]), &live, &run), "KDE: DP-1 HDR off");
        assert_eq!(*asked.borrow(), ["gdbus", "kscreen-doctor"]);
    }
}
