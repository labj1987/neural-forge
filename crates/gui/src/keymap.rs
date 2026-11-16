//! The session's keyboard layout, for the toggle key: the layer reads the key from `/dev/input` by its Linux evdev
//! code (the key's place on the keyboard), so the row shows a stored code by the name the key carries on the user's
//! layout, and can find the code of a key the toolkit only names.

use xkbcommon::xkb;

/// The difference between an X/xkb keycode and the evdev code of the same key.
const EVDEV_OFFSET: u32 = 8;

/// The highest evdev key code (`KEY_MAX`).
const EVDEV_MAX: u32 = 767;

/// A keymap and the layout of it that is in use (0 where that is not known).
pub struct SessionKeys {
    keymap: Option<xkb::Keymap>,
    layout: u32,
}

impl SessionKeys {
    /// From a keymap as the compositor hands it out (xkb text format).
    pub fn from_text(text: &str, layout: u32) -> Self {
        let context = xkb::Context::new(xkb::CONTEXT_NO_FLAGS);
        let keymap = xkb::Keymap::new_from_string(&context, text.to_string(), xkb::KEYMAP_FORMAT_TEXT_V1, xkb::KEYMAP_COMPILE_NO_FLAGS);
        Self { keymap, layout }
    }

    /// From the names a layout goes by (rules, model, layout, variant, options), as an X server publishes them. Empty
    /// names are xkbcommon's defaults: `XKB_DEFAULT_*` in the environment, else the system's.
    pub fn from_names(rules: &str, model: &str, layout: &str, variant: &str, options: &str) -> Self {
        let context = xkb::Context::new(xkb::CONTEXT_NO_FLAGS);
        let options = (!options.is_empty()).then(|| options.to_string());
        let keymap = xkb::Keymap::new_from_names(&context, rules, model, layout, variant, options, xkb::KEYMAP_COMPILE_NO_FLAGS);
        Self { keymap, layout: 0 }
    }

    /// The same keymap read in its layout number `layout` (the one in use, of several).
    pub fn in_layout(mut self, layout: u32) -> Self {
        self.layout = layout;
        self
    }

    /// Which layout of the keymap was in use when the key with evdev code `code` gave what the toolkit calls `key`:
    /// the layout already taken to be in use if the key gives that there, else the first that does. `None` when no
    /// layout's key gives it (or all of them would: nothing learnt). For a compositor that does not say which layout
    /// is in use.
    pub fn layout_of_press(&self, code: u32, key: &str) -> Option<u32> {
        let keymap = self.keymap.as_ref()?;
        let keycode = xkb::Keycode::new(code + EVDEV_OFFSET);
        let key = key.to_lowercase();
        let gives = |layout: u32| {
            (0..keymap.num_levels_for_key(keycode, layout))
                .any(|level| keymap.key_get_syms_by_level(keycode, layout, level).iter().any(|&symbol| toolkit_key_name(symbol) == key))
        };
        let layouts = keymap.num_layouts_for_key(keycode);
        if layouts < 2 || (0..layouts).all(gives) {
            return None;
        }
        if self.layout < layouts && gives(self.layout) {
            return Some(self.layout);
        }
        (0..layouts).find(|&layout| gives(layout))
    }

    /// The layout xkbcommon takes by default, where the session's own keymap could not be had.
    pub fn default_layout() -> Self {
        Self::from_names("", "", "", "", "")
    }

    /// Whether a keymap was compiled at all (without one every key is "Key N" and no name finds a code).
    pub fn known(&self) -> bool {
        self.keymap.is_some()
    }

    /// The layout to read `keycode` in: the one in use, or the key's first where it has fewer.
    fn layout_for(&self, keymap: &xkb::Keymap, keycode: xkb::Keycode) -> u32 {
        if self.layout < keymap.num_layouts_for_key(keycode) {
            self.layout
        } else {
            0
        }
    }

    /// The name of the key with evdev code `code` on this layout ("F11", "Scroll_Lock", "A"; "KP_End" for the number
    /// pad's 1): what the key gives unshifted. "Unbound" for 0, "Key N" for a code the layout has no key for.
    pub fn label(&self, code: u32) -> String {
        if code == 0 {
            return "Unbound".to_owned();
        }
        let name = self.keymap.as_ref().and_then(|keymap| {
            let keycode = xkb::Keycode::new(code + EVDEV_OFFSET);
            let symbols = keymap.key_get_syms_by_level(keycode, self.layout_for(keymap, keycode), 0);
            symbols.first().map(|&symbol| xkb::keysym_get_name(symbol))
        });
        crate::text::format_key_label(code, name.as_deref())
    }

    /// The evdev code of the key the toolkit calls `key`, on this layout: the key that gives it unshifted in the layout
    /// in use before a shifted one or another layout's, and a main-block key before the number pad's (the toolkit
    /// names both alike). `None` when no key of the layout gives it.
    pub fn code_for(&self, key: &str) -> Option<u32> {
        let keymap = self.keymap.as_ref()?;
        let key = key.to_lowercase();
        let mut best: Option<((bool, bool, u32, u32), u32)> = None;
        for raw in keymap.min_keycode().raw()..=keymap.max_keycode().raw() {
            let Some(code) = raw.checked_sub(EVDEV_OFFSET).filter(|code| (1..=EVDEV_MAX).contains(code)) else { continue };
            let keycode = xkb::Keycode::new(raw);
            for layout in 0..keymap.num_layouts_for_key(keycode) {
                for level in 0..keymap.num_levels_for_key(keycode, layout) {
                    for &symbol in keymap.key_get_syms_by_level(keycode, layout, level) {
                        if toolkit_key_name(symbol) != key {
                            continue;
                        }
                        let rank = (symbol.is_keypad_key(), layout != self.layout_for(keymap, keycode), level, layout);
                        if best.is_none_or(|(held, _)| rank < held) {
                            best = Some((rank, code));
                        }
                    }
                }
            }
        }
        best.map(|(_, code)| code)
    }
}

/// The name the toolkit gives a key that produces `symbol` (its `Keystroke::key`): the character for a printable
/// ASCII one, a few names of its own, else xkb's name in lower case, the number pad's without its `KP_`.
fn toolkit_key_name(symbol: xkb::Keysym) -> String {
    use xkb::Keysym;
    let named = match symbol {
        Keysym::Return => "enter",
        Keysym::Prior | Keysym::KP_Prior => "pageup",
        Keysym::Next | Keysym::KP_Next => "pagedown",
        Keysym::ISO_Left_Tab | Keysym::Tab => "tab",
        Keysym::space => "space",
        Keysym::BackSpace => "backspace",
        Keysym::Delete => "delete",
        Keysym::Escape => "escape",
        _ => "",
    };
    if !named.is_empty() {
        return named.to_owned();
    }
    let name = xkb::keysym_get_name(symbol).to_lowercase();
    if symbol.is_keypad_key() {
        return name.replace("kp_", "");
    }
    let text = xkb::keysym_to_utf8(symbol);
    let text = text.trim_end_matches('\0');
    match text.chars().next() {
        Some(c) if text.len() == 1 && c.is_ascii_graphic() => text.to_lowercase(),
        _ => name,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(name: &str) -> SessionKeys {
        let keys = SessionKeys::from_names("evdev", "pc105", name, "", "");
        assert!(keys.known(), "the {name} layout compiles (xkeyboard-config is installed)");
        keys
    }

    // Linux evdev codes (input-event-codes.h).
    const KEY_1: u32 = 2;
    const KEY_Q: u32 = 16;
    const KEY_W: u32 = 17;
    const KEY_Y: u32 = 21;
    const KEY_A: u32 = 30;
    const KEY_Z: u32 = 44;
    const KEY_SCROLLLOCK: u32 = 70;
    const KEY_KP1: u32 = 79;
    const KEY_F11: u32 = 87;
    const KEY_KPENTER: u32 = 96;
    const KEY_END: u32 = 107;

    #[test]
    fn a_named_key_is_the_key_that_carries_the_name_on_the_users_layout() {
        let us = layout("us");
        assert_eq!(us.code_for("a"), Some(KEY_A));
        assert_eq!(us.code_for("q"), Some(KEY_Q));
        assert_eq!(us.code_for("1"), Some(KEY_1), "the main block's 1, not the number pad's");
        // AZERTY: A and Q have changed places, and the digit row gives its digits only with Shift.
        let fr = layout("fr");
        assert_eq!(fr.code_for("a"), Some(KEY_Q));
        assert_eq!(fr.code_for("q"), Some(KEY_A));
        assert_eq!(fr.code_for("z"), Some(KEY_W));
        assert_eq!(fr.code_for("&"), Some(KEY_1), "what the digit row's first key gives unshifted");
        assert_eq!(fr.code_for("1"), Some(KEY_1), "and its shifted digit is the same key");
        // QWERTZ: Y and Z have changed places.
        let de = layout("de");
        assert_eq!(de.code_for("z"), Some(KEY_Y));
        assert_eq!(de.code_for("y"), Some(KEY_Z));
        // The same on every layout.
        for keys in [&us, &fr, &de] {
            assert_eq!(keys.code_for("f11"), Some(KEY_F11));
            assert_eq!(keys.code_for("scroll_lock"), Some(KEY_SCROLLLOCK));
            assert_eq!(keys.code_for("end"), Some(KEY_END), "the navigation block's End, not the number pad's");
            assert_eq!(keys.code_for("no such key"), None);
        }
    }

    #[test]
    fn a_stored_code_is_shown_by_the_keys_name_on_the_users_layout() {
        let (us, fr, de) = (layout("us"), layout("fr"), layout("de"));
        assert_eq!(us.label(KEY_Q), "Q");
        assert_eq!(fr.label(KEY_Q), "A", "the key in Q's place is A on AZERTY");
        assert_eq!(de.label(KEY_Y), "Z");
        assert_eq!(fr.label(KEY_1), "ampersand");
        for keys in [&us, &fr, &de] {
            assert_eq!(keys.label(KEY_F11), "F11");
            assert_eq!(keys.label(KEY_SCROLLLOCK), "Scroll_Lock");
            assert_eq!(keys.label(KEY_KP1), "KP_End", "the number pad's keys are told from the main block's");
            assert_eq!(keys.label(KEY_KPENTER), "KP_Enter");
            assert_eq!(keys.label(0), "Unbound");
            assert_eq!(keys.label(766), "Key 766", "a code the layout has no key for");
        }
    }

    #[test]
    fn names_follow_the_layout_in_use_of_several() {
        let both = SessionKeys::from_names("evdev", "pc105", "us,fr", "", "");
        assert_eq!(both.label(KEY_Q), "Q");
        assert_eq!(both.code_for("a"), Some(KEY_A));
        let french = both.in_layout(1);
        assert_eq!(french.label(KEY_Q), "A", "the second layout is the one in use");
        assert_eq!(french.code_for("a"), Some(KEY_Q));
        assert_eq!(french.in_layout(7).label(KEY_Q), "Q", "a layout the keymap does not have reads as its first");
    }

    #[test]
    fn a_press_tells_which_layout_is_in_use() {
        let both = SessionKeys::from_names("evdev", "pc105", "us,fr", "", "");
        assert_eq!(both.layout_of_press(KEY_Q, "a"), Some(1), "the key in Q's place gave an a: the French layout");
        assert_eq!(both.layout_of_press(KEY_1, "&"), Some(1));
        assert_eq!(both.layout_of_press(KEY_Q, "q"), Some(0));
        assert_eq!(both.in_layout(1).layout_of_press(KEY_A, "q"), Some(1), "the layout taken to be in use stays while it fits");
        let both = SessionKeys::from_names("evdev", "pc105", "us,fr", "", "");
        assert_eq!(both.layout_of_press(KEY_F11, "f11"), None, "a key that is the same on both says nothing");
        assert_eq!(layout("fr").layout_of_press(KEY_Q, "a"), None, "one layout: nothing to tell");
    }

    #[test]
    fn a_keymap_as_a_compositor_sends_it_reads_the_same() {
        let context = xkb::Context::new(xkb::CONTEXT_NO_FLAGS);
        let fr = xkb::Keymap::new_from_names(&context, "evdev", "pc105", "fr", "", None, xkb::KEYMAP_COMPILE_NO_FLAGS).unwrap();
        let keys = SessionKeys::from_text(&fr.get_as_string(xkb::KEYMAP_FORMAT_TEXT_V1), 0);
        assert_eq!(keys.code_for("a"), Some(KEY_Q));
        assert_eq!(keys.label(KEY_Q), "A");
        // Two layouts in one keymap, the second in use: its names, and the first's only for what it lacks.
        let both = xkb::Keymap::new_from_names(&context, "evdev", "pc105", "us,fr", "", None, xkb::KEYMAP_COMPILE_NO_FLAGS).unwrap();
        let keys = SessionKeys::from_text(&both.get_as_string(xkb::KEYMAP_FORMAT_TEXT_V1), 1);
        assert_eq!(keys.code_for("a"), Some(KEY_Q));
        assert_eq!(keys.label(KEY_A), "Q");
        assert!(!SessionKeys::from_text("not a keymap", 0).known());
        assert_eq!(SessionKeys::from_text("not a keymap", 0).label(KEY_F11), "Key 87");
    }
}
