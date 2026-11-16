//! Keys for the toggle key row: the layer reads the key from `/dev/input` by its Linux evdev code, and the toolkit
//! reports a pressed key by its name on the current layout, not by its code. The table gives the code of the key
//! that carries that name on a US layout (function, navigation and lock keys are in the same place on every layout).

/// (evdev code, the toolkit's key name, the name shown).
const KEYS: &[(u32, &str, &str)] = &[
    (2, "1", "1"),
    (3, "2", "2"),
    (4, "3", "3"),
    (5, "4", "4"),
    (6, "5", "5"),
    (7, "6", "6"),
    (8, "7", "7"),
    (9, "8", "8"),
    (10, "9", "9"),
    (11, "0", "0"),
    (12, "-", "minus"),
    (13, "=", "equal"),
    (14, "backspace", "BackSpace"),
    (15, "tab", "Tab"),
    (16, "q", "Q"),
    (17, "w", "W"),
    (18, "e", "E"),
    (19, "r", "R"),
    (20, "t", "T"),
    (21, "y", "Y"),
    (22, "u", "U"),
    (23, "i", "I"),
    (24, "o", "O"),
    (25, "p", "P"),
    (26, "[", "bracketleft"),
    (27, "]", "bracketright"),
    (28, "enter", "Return"),
    (30, "a", "A"),
    (31, "s", "S"),
    (32, "d", "D"),
    (33, "f", "F"),
    (34, "g", "G"),
    (35, "h", "H"),
    (36, "j", "J"),
    (37, "k", "K"),
    (38, "l", "L"),
    (39, ";", "semicolon"),
    (40, "'", "apostrophe"),
    (41, "`", "grave"),
    (43, "\\", "backslash"),
    (44, "z", "Z"),
    (45, "x", "X"),
    (46, "c", "C"),
    (47, "v", "V"),
    (48, "b", "B"),
    (49, "n", "N"),
    (50, "m", "M"),
    (51, ",", "comma"),
    (52, ".", "period"),
    (53, "/", "slash"),
    (57, "space", "space"),
    (58, "caps_lock", "Caps_Lock"),
    (59, "f1", "F1"),
    (60, "f2", "F2"),
    (61, "f3", "F3"),
    (62, "f4", "F4"),
    (63, "f5", "F5"),
    (64, "f6", "F6"),
    (65, "f7", "F7"),
    (66, "f8", "F8"),
    (67, "f9", "F9"),
    (68, "f10", "F10"),
    (69, "num_lock", "Num_Lock"),
    (70, "scroll_lock", "Scroll_Lock"),
    (87, "f11", "F11"),
    (88, "f12", "F12"),
    (99, "print", "Print"),
    (102, "home", "Home"),
    (103, "up", "Up"),
    (104, "pageup", "Page_Up"),
    (105, "left", "Left"),
    (106, "right", "Right"),
    (107, "end", "End"),
    (108, "down", "Down"),
    (109, "pagedown", "Page_Down"),
    (110, "insert", "Insert"),
    (111, "delete", "Delete"),
    (119, "pause", "Pause"),
    (127, "menu", "Menu"),
    (183, "f13", "F13"),
    (184, "f14", "F14"),
    (185, "f15", "F15"),
    (186, "f16", "F16"),
    (187, "f17", "F17"),
    (188, "f18", "F18"),
    (189, "f19", "F19"),
    (190, "f20", "F20"),
    (191, "f21", "F21"),
    (192, "f22", "F22"),
    (193, "f23", "F23"),
    (194, "f24", "F24"),
];

/// The evdev code of the key the toolkit names `key` (its names are lower case), if it is one the toggle can use.
pub fn evdev_code(key: &str) -> Option<u32> {
    let key = key.to_ascii_lowercase();
    let key = match key.as_str() {
        "prior" | "page_up" => "pageup",
        "next" | "page_down" => "pagedown",
        "return" => "enter",
        "sys_req" | "printscreen" => "print",
        other => other,
    };
    KEYS.iter().find(|&&(_, name, _)| name == key).map(|&(code, _, _)| code)
}

/// The key's name for an evdev code ("F11", "Scroll_Lock", "A"), "Key N" for one the table does not have.
pub fn key_label(code: u32) -> String {
    if code == 0 {
        return "Unbound".to_owned();
    }
    crate::text::format_key_label(code, KEYS.iter().find(|&&(c, _, _)| c == code).map(|&(_, _, shown)| shown))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pressed_key_gives_the_code_the_layer_reads() {
        assert_eq!(evdev_code("f11"), Some(87));
        assert_eq!(evdev_code("F11"), Some(87));
        assert_eq!(evdev_code("a"), Some(30));
        assert_eq!(evdev_code("scroll_lock"), Some(70));
        assert_eq!(evdev_code("Prior"), Some(104));
        assert_eq!(evdev_code("shift"), None, "a modifier alone is not a toggle key");
        assert_eq!(evdev_code(""), None);
    }

    #[test]
    fn every_code_is_listed_once_and_shown_by_name() {
        for (i, &(code, name, _)) in KEYS.iter().enumerate() {
            assert!(code > 0 && code <= 767);
            assert!(KEYS[i + 1..].iter().all(|&(c, n, _)| c != code && n != name), "{name} twice");
            assert_eq!(evdev_code(name), Some(code));
        }
        assert_eq!(key_label(0), "Unbound");
        assert_eq!(key_label(87), "F11");
        assert_eq!(key_label(30), "A");
        assert_eq!(key_label(700), "Key 700");
    }
}
