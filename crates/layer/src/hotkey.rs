//! Minimal X11 keyboard polling for the in-game NR toggle.  The game runs through
//! XWayland on the supported desktop, so its Vulkan process has the same X display
//! access as the game window without requiring input-device permissions.

use std::ffi::c_char;

#[repr(C)]
struct Display { _private: [u8; 0] }

#[link(name = "X11")]
unsafe extern "C" {
    fn XOpenDisplay(name: *const c_char) -> *mut Display;
    fn XQueryKeymap(display: *mut Display, keys_return: *mut c_char) -> i32;
    fn XCloseDisplay(display: *mut Display) -> i32;
}

/// `XQueryKeymap` is a synchronous round trip to the X server, so it is not done every
/// frame: at most once per this interval. Short enough that a normal key tap (well over
/// 50 ms) is always seen, whatever the frame rate.
const POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(50);

pub struct Poller {
    display: *mut Display,
    was_pressed: bool,
    last_poll: Option<std::time::Instant>,
}

// The display connection is used only from the layer's serialized present hook.
unsafe impl Send for Poller {}

impl Default for Poller {
    fn default() -> Self { Self { display: std::ptr::null_mut(), was_pressed: false, last_poll: None } }
}

impl Poller {
    /// Returns true only on a press edge. `evdev_code` is what the GUI persists;
    /// X11 keycodes are offset by eight on Xwayland/XKB.
    pub fn pressed(&mut self, evdev_code: u32) -> bool {
        if evdev_code == 0 || evdev_code > 247 { self.was_pressed = false; return false; }
        let now = std::time::Instant::now();
        if self.last_poll.is_some_and(|last| now.duration_since(last) < POLL_INTERVAL) { return false; }
        self.last_poll = Some(now);
        if self.display.is_null() {
            // SAFETY: null selects the process's DISPLAY; failure simply disables the
            // hotkey (for example on a headless Vulkan application).
            self.display = unsafe { XOpenDisplay(std::ptr::null()) };
            if self.display.is_null() { return false; }
        }
        let mut keys = [0i8; 32];
        // SAFETY: `display` is live after XOpenDisplay and keys has the required 32 bytes.
        unsafe { XQueryKeymap(self.display, keys.as_mut_ptr()); }
        let keycode = (evdev_code + 8) as usize;
        let pressed = (keys[keycode / 8] as u8 & (1 << (keycode % 8))) != 0;
        let edge = pressed && !self.was_pressed;
        self.was_pressed = pressed;
        edge
    }
}

impl Drop for Poller {
    fn drop(&mut self) {
        if !self.display.is_null() {
            // SAFETY: `display` came from `XOpenDisplay`, is closed exactly once here, and
            // the poller (its only user) is being destroyed.
            unsafe { XCloseDisplay(self.display); }
            self.display = std::ptr::null_mut();
        }
    }
}
