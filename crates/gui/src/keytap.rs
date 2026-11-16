//! The session's keymap and, where the display server will say, the pressed key's hardware code, for the toggle key
//! row.
//!
//! The toolkit's key events name a key (by what it gives on the layout) but do not say which key it was, and the
//! layer reads keys from `/dev/input` by their place on the keyboard.
//!
//! - X11: a second, private connection selects XInput 2 raw key presses on the root window while the row waits for a
//!   key (their detail is the keycode, so the number pad's keys are told from the main block's); the layout's names
//!   are the root window's `_XKB_RULES_NAMES`.
//! - Wayland: a private connection of its own asks the compositor for a `wl_keyboard` only to be sent the keymap,
//!   and the pressed key is the one that carries the toolkit's name for it on that keymap (so the number pad's keys
//!   count as their main-block twins there). Only after the seat has said it has a keyboard: a headless GNOME Shell 50
//!   crashed in libmutter-mtk when asked for the keyboard of a seat without one.

use std::os::fd::AsRawFd;

use gpui_kit::Window;
use raw_window_handle::{HasDisplayHandle, RawDisplayHandle};
use wayland_client::globals::{registry_queue_init, GlobalListContents};
use wayland_client::protocol::wl_keyboard::{self, WlKeyboard};
use wayland_client::protocol::wl_registry::WlRegistry;
use wayland_client::protocol::wl_seat::{self, WlSeat};
use wayland_client::{Connection, Dispatch, Proxy, QueueHandle, WEnum};
use x11rb::connection::Connection as _;
use x11rb::protocol::xinput::{self, ConnectionExt as _};
use x11rb::protocol::xproto::{AtomEnum, ConnectionExt as _};
use x11rb::protocol::Event;
use x11rb::rust_connection::RustConnection;

use crate::keymap::SessionKeys;

type Failure = Box<dyn std::error::Error>;

/// The session's keymap, from whichever display server `window` is on. `None` where it cannot be had (the caller
/// keeps xkbcommon's default layout).
pub fn session_keys(window: &Window) -> Option<SessionKeys> {
    let display = HasDisplayHandle::display_handle(window).ok()?;
    let keys = match display.as_raw() {
        RawDisplayHandle::Wayland(_) => wayland_keymap().map(|text| SessionKeys::from_text(&text, 0)),
        RawDisplayHandle::Xcb(_) | RawDisplayHandle::Xlib(_) => X11Tap::start().and_then(|tap| tap.keys().ok_or_else(|| "the X server names no layout".into())),
        _ => return None,
    };
    keys.inspect_err(|e| log::info!("the toggle key is named by the default layout: the session's keymap could not be read ({e})"))
        .ok()
        .filter(SessionKeys::known)
}

/// A second ear on the X server's keyboard, for as long as it lives.
pub struct KeyTap(Box<X11Tap>);

impl KeyTap {
    /// Starts listening where `window`'s display server reports raw key presses (X11). `None` elsewhere, and where
    /// it cannot be done (the reason logged): the row then goes by the key's name.
    pub fn start(window: &Window) -> Option<Self> {
        let display = HasDisplayHandle::display_handle(window).ok()?;
        if !matches!(display.as_raw(), RawDisplayHandle::Xcb(_) | RawDisplayHandle::Xlib(_)) {
            return None;
        }
        X11Tap::start().inspect_err(|e| log::info!("the toggle key goes by key names: no raw key events ({e})")).ok().map(|tap| Self(Box::new(tap)))
    }

    /// The evdev code of the key pressed last since the tap started, if the X server reported one.
    pub fn last_press(&mut self) -> Option<u32> {
        self.0.last_press()
    }
}

// ── Wayland ──────────────────────────────────────────────────────────────────

#[derive(Default)]
struct TapState {
    has_keyboard: bool,
    keymap: Option<String>,
}

/// The compositor's keymap (xkb text), asked for on a connection of this function's own, which is closed again.
fn wayland_keymap() -> Result<String, Failure> {
    let connection = Connection::connect_to_env()?;
    let (globals, mut queue) = registry_queue_init::<TapState>(&connection)?;
    let handle = queue.handle();
    let seat: WlSeat = globals.bind(&handle, 1..=5, ())?;
    let mut state = TapState::default();
    // The seat says first what it has: asking a seat without a keyboard for one is a protocol violation (and a
    // headless GNOME Shell 50 crashed on it instead of refusing).
    queue.roundtrip(&mut state)?;
    if !state.has_keyboard {
        return Err("the seat has no keyboard".into());
    }
    let keyboard = seat.get_keyboard(&handle, ());
    // The keymap comes with the keyboard.
    queue.roundtrip(&mut state)?;
    if keyboard.version() >= 3 {
        keyboard.release();
    }
    if seat.version() >= 5 {
        seat.release();
    }
    let _ = connection.flush();
    state.keymap.ok_or_else(|| "the compositor sent no keymap".into())
}

impl Dispatch<WlRegistry, GlobalListContents> for TapState {
    fn event(_: &mut Self, _: &WlRegistry, _: <WlRegistry as Proxy>::Event, _: &GlobalListContents, _: &Connection, _: &QueueHandle<Self>) {}
}

impl Dispatch<WlSeat, ()> for TapState {
    fn event(state: &mut Self, _: &WlSeat, event: wl_seat::Event, (): &(), _: &Connection, _: &QueueHandle<Self>) {
        if let wl_seat::Event::Capabilities { capabilities: WEnum::Value(capabilities) } = event {
            state.has_keyboard = capabilities.contains(wl_seat::Capability::Keyboard);
        }
    }
}

impl Dispatch<WlKeyboard, ()> for TapState {
    fn event(state: &mut Self, _: &WlKeyboard, event: wl_keyboard::Event, (): &(), _: &Connection, _: &QueueHandle<Self>) {
        if let wl_keyboard::Event::Keymap { format: WEnum::Value(wl_keyboard::KeymapFormat::XkbV1), fd, size } = event {
            state.keymap = read_keymap(&fd, size as usize);
        }
    }
}

/// The keymap text behind the file descriptor a compositor sends (it must be mapped, privately, to be read).
fn read_keymap(fd: &impl AsRawFd, size: usize) -> Option<String> {
    if size == 0 {
        return None;
    }
    // SAFETY: a private read-only mapping of `size` bytes of the descriptor, checked for failure, read as bytes
    // within its length and unmapped before returning.
    unsafe {
        let map = libc::mmap(std::ptr::null_mut(), size, libc::PROT_READ, libc::MAP_PRIVATE, fd.as_raw_fd(), 0);
        if map == libc::MAP_FAILED {
            return None;
        }
        let bytes = std::slice::from_raw_parts(map.cast::<u8>(), size);
        let text = String::from_utf8_lossy(bytes.split(|&b| b == 0).next().unwrap_or_default()).into_owned();
        libc::munmap(map, size);
        Some(text)
    }
}

// ── X11 ──────────────────────────────────────────────────────────────────────

pub struct X11Tap {
    connection: RustConnection,
    root: u32,
}

/// How long a raw key press may take to arrive on this connection after the toolkit got the key on its own.
const RAW_KEY_WAIT: std::time::Duration = std::time::Duration::from_millis(60);

impl X11Tap {
    fn start() -> Result<Self, Failure> {
        let (connection, screen) = x11rb::connect(None)?;
        let root = connection.setup().roots.get(screen).ok_or("the X server names no such screen")?.root;
        connection.xinput_xi_query_version(2, 0)?.reply()?;
        let mask = xinput::EventMask { deviceid: u16::from(xinput::Device::ALL_MASTER), mask: vec![xinput::XIEventMask::RAW_KEY_PRESS] };
        connection.xinput_xi_select_events(root, &[mask])?.check()?;
        Ok(Self { connection, root })
    }

    /// The layout the X server names (`_XKB_RULES_NAMES`: rules, model, layout, variant, options).
    fn keys(&self) -> Option<SessionKeys> {
        let atom = self.connection.intern_atom(true, b"_XKB_RULES_NAMES").ok()?.reply().ok()?.atom;
        let property = self.connection.get_property(false, self.root, atom, AtomEnum::STRING, 0, 1024).ok()?.reply().ok()?;
        let names: Vec<String> = property.value.split(|&b| b == 0).map(|name| String::from_utf8_lossy(name).into_owned()).collect();
        let name = |i: usize| names.get(i).map_or("", String::as_str);
        (!name(2).is_empty()).then(|| SessionKeys::from_names(name(0), name(1), name(2), name(3), name(4)))
    }

    fn last_press(&mut self) -> Option<u32> {
        let mut pressed = None;
        let deadline = std::time::Instant::now() + RAW_KEY_WAIT;
        loop {
            while let Ok(Some(event)) = self.connection.poll_for_event() {
                if let Event::XinputRawKeyPress(press) = event {
                    // An X keycode is the evdev code plus 8.
                    pressed = press.detail.checked_sub(8).filter(|&code| code > 0);
                }
            }
            if pressed.is_some() || std::time::Instant::now() >= deadline {
                return pressed;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }
}
