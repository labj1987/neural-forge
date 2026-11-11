//! The desktop's own HDR state, for the HDR verdict (the layer's `hdr_verdict` says what the game asked for; this
//! says whether the screen it goes to is in HDR). Read without root and without changing anything: GNOME's
//! `org.gnome.Mutter.DisplayConfig.GetCurrentState` over the session bus (`gdbus`; each monitor's `color-mode`, 1
//! being BT.2100), KDE Plasma's `kscreen-doctor -j` (each output's `hdr`), and a gamescope session from its
//! environment. `None` where none of them answers.

use crate::doctor::{run_command, CommandOutput};

const TIMEOUT: std::time::Duration = std::time::Duration::from_secs(3);

/// One display as the compositor describes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Output {
    pub name: String,
    pub hdr: bool,
}

/// "GNOME: HDMI-1 HDR on (BT.2100)", "KDE: DP-1 HDR on, HDMI-A-1 HDR off", "gamescope session (...)", or `None`.
pub fn read() -> Option<String> {
    if std::env::var_os("GAMESCOPE_WAYLAND_DISPLAY").is_some() {
        return Some("gamescope session: HDR is gamescope's (--hdr-enabled) and the display's".into());
    }
    let gnome = run_command(
        "gdbus",
        &[
            "call",
            "--session",
            "--dest",
            "org.gnome.Mutter.DisplayConfig",
            "--object-path",
            "/org/gnome/Mutter/DisplayConfig",
            "--method",
            "org.gnome.Mutter.DisplayConfig.GetCurrentState",
        ],
        TIMEOUT,
    );
    if let CommandOutput::Ran(text) = gnome {
        let outputs = parse_mutter(&text);
        if !outputs.is_empty() {
            return Some(describe("GNOME", &outputs, " (BT.2100)"));
        }
    }
    if let CommandOutput::Ran(text) = run_command("kscreen-doctor", &["-j"], TIMEOUT) {
        let outputs = parse_kscreen(&text);
        if !outputs.is_empty() {
            return Some(describe("KDE", &outputs, ""));
        }
    }
    None
}

fn describe(desktop: &str, outputs: &[Output], on_detail: &str) -> String {
    let list: Vec<String> = outputs.iter().map(|o| format!("{} HDR {}", o.name, if o.hdr { format!("on{on_detail}") } else { "off".into() })).collect();
    format!("{desktop}: {}", list.join(", "))
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
            out.push(Output { name: name.to_string(), hdr: mode == 1 });
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
        .filter_map(|o| Some(Output { name: o.get("name")?.as_str()?.to_string(), hdr: o.get("hdr").and_then(|h| h.as_bool()).unwrap_or(false) }))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mutter_monitors_are_read_with_their_colour_mode() {
        // Shortened from GNOME 49's reply on the test machine.
        let text = "(uint32 1, [(('HDMI-1', 'TCL', 'Beyond TV', '0x00010000'), [('4096x2160@59.940', 4096, 2160, 59.94, 1.0, [1.0], @a{sv} {})], \
                    {'is-builtin': <false>, 'display-name': <'Technical Concepts Ltd 55\"'>, 'color-mode': <uint32 1>, 'supported-color-modes': <[uint32 0, 2, 1]>}), \
                    (('DP-2', 'DEL', 'U2720Q', 'X'), [('3840x2160@60.000', 3840, 2160, 60.0, 1.0, [1.0], {})], {'color-mode': <uint32 0>})], \
                    [(0, 0, 1.0, uint32 0, true, [('HDMI-1', 'TCL', 'Beyond TV', '0x00010000')], @a{sv} {})], {'layout-mode': <uint32 1>})";
        let outputs = parse_mutter(text);
        assert_eq!(outputs, vec![Output { name: "HDMI-1".into(), hdr: true }, Output { name: "DP-2".into(), hdr: false }]);
        assert_eq!(describe("GNOME", &outputs, " (BT.2100)"), "GNOME: HDMI-1 HDR on (BT.2100), DP-2 HDR off");
        assert!(parse_mutter("Error: GDBus.Error:org.freedesktop.DBus.Error.ServiceUnknown").is_empty());
    }

    #[test]
    fn kscreen_outputs_are_read_with_their_hdr_flag() {
        let text = r#"{"outputs": [{"name": "DP-1", "enabled": true, "hdr": true}, {"name": "HDMI-A-1", "enabled": true}, {"name": "eDP-1", "enabled": false, "hdr": true}]}"#;
        assert_eq!(parse_kscreen(text), vec![Output { name: "DP-1".into(), hdr: true }, Output { name: "HDMI-A-1".into(), hdr: false }]);
        assert!(parse_kscreen("not json").is_empty());
    }
}
