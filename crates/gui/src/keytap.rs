//! The pressed key's hardware code and the session's keymap, for the toggle key row.
//!
//! The toolkit's key events name a key (by what it gives on the layout) but do not say which key it was, and the
//! layer reads keys from `/dev/input` by their place on the keyboard. So while the row waits for a key, the display
//! server is asked for the same key press directly:
//!
//! - Wayland: a second `wl_keyboard` on the toolkit's own connection (a compositor sends a window's key presses to
//!   every keyboard object of its client, with the evdev code), which also brings the keymap and the layout in use.
//!   For the key's name at start, before any key is waited for, a private connection asks for the keymap alone.
//!   Either way only after the seat has said it has a keyboard: a headless GNOME Shell 50 crashed in libmutter-mtk
//!   when asked for the keyboard of a seat without one.
//! - X11: a second, private connection selects XInput 2 raw key presses on the root window (their detail is the
//!   keycode); the layout's names are the root window's `_XKB_RULES_NAMES`.
//!
//! Where no direct report comes, the row falls back to the key that carries the toolkit's name for the press on the
//! session's keymap (`SessionKeys::code_for`), which cannot tell the number pad's keys from the main block's.

use std::os::fd::AsRawFd;

use gpui_kit::Window;
use raw_window_handle::{HasDisplayHandle, RawDisplayHandle};
use wayland_client::backend::Backend;
use wayland_client::globals::{registry_queue_init, GlobalListContents};
use wayland_client::protocol::wl_keyboard::{self, WlKeyboard};
use wayland_client::protocol::wl_registry::WlRegistry;
use wayland_client::protocol::wl_seat::{self, WlSeat};
use wayland_client::{Connection, Dispatch, EventQueue, Proxy, QueueHandle, WEnum};
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

/// A second ear on the display server's keyboard, for as long as it lives.
pub enum KeyTap {
    Wayland(WaylandTap),
    X11(Box<X11Tap>),
}

impl KeyTap {
    /// Starts listening on `window`'s display server. `None` where that cannot be done (the reason logged): the row
    /// then goes by the key's name.
    pub fn start(window: &Window) -> Option<Self> {
        let display = HasDisplayHandle::display_handle(window).ok()?;
        let started = match display.as_raw() {
            // SAFETY: the pointer is the live window's wl_display, this runs on the UI thread, which owns the
            // connection, and the tap lives in the window's view, so it is dropped while the display is open.
            RawDisplayHandle::Wayland(display) => unsafe { WaylandTap::start(display.display.as_ptr()) }.map(KeyTap::Wayland),
            RawDisplayHandle::Xcb(_) | RawDisplayHandle::Xlib(_) => X11Tap::start().map(|tap| KeyTap::X11(Box::new(tap))),
            _ => return None,
        };
        started.inspect_err(|e| log::info!("the toggle key goes by key names: no direct key events ({e})")).ok()
    }

    /// The session's keymap as this tap got it, with the layout in use where the display server said.
    pub fn keys(&self) -> Option<SessionKeys> {
        match self {
            KeyTap::Wayland(tap) => tap.state.keymap.as_deref().map(|text| SessionKeys::from_text(text, tap.state.layout)),
            KeyTap::X11(tap) => tap.keys(),
        }
        .filter(SessionKeys::known)
    }

    /// The evdev code of the key pressed last since the tap started, if the display server reported one.
    pub fn last_press(&mut self) -> Option<u32> {
        match self {
            KeyTap::Wayland(tap) => tap.last_press(),
            KeyTap::X11(tap) => tap.last_press(),
        }
    }
}

// ── Wayland ──────────────────────────────────────────────────────────────────

#[derive(Default)]
struct TapState {
    has_keyboard: bool,
    keymap: Option<String>,
    /// The layout in use (the keyboard's group), as the last modifiers event said.
    layout: u32,
    pressed: Option<u32>,
}

/// A second keyboard object on the toolkit's own Wayland connection.
pub struct WaylandTap {
    queue: EventQueue<TapState>,
    state: TapState,
    connection: Connection,
    seat: WlSeat,
    keyboard: WlKeyboard,
}

impl WaylandTap {
    /// # Safety
    /// `display` must be a live `wl_display`, no other thread may be reading the connection, and the tap must be
    /// dropped while the display is still open.
    unsafe fn start(display: *mut std::ffi::c_void) -> Result<Self, Failure> {
        // A guest handle on the toolkit's connection: dropping it closes nothing.
        let connection = Connection::from_backend(Backend::from_foreign_display(display.cast()));
        let (globals, mut queue) = registry_queue_init::<TapState>(&connection)?;
        let handle = queue.handle();
        let seat: WlSeat = globals.bind(&handle, 1..=5, ())?;
        let mut state = TapState::default();
        // The seat says first what it has (see the module's note).
        queue.roundtrip(&mut state)?;
        if !state.has_keyboard {
            if seat.version() >= 5 {
                seat.release();
            }
            let _ = connection.flush();
            return Err("the seat has no keyboard".into());
        }
        let keyboard = seat.get_keyboard(&handle, ());
        // The keymap comes with the keyboard, and for the window that has the focus the layout in use.
        queue.roundtrip(&mut state)?;
        state.pressed = None;
        Ok(Self { queue, state, connection, seat, keyboard })
    }

    fn last_press(&mut self) -> Option<u32> {
        // The toolkit has read the socket to deliver its own key event; this keyboard's copy waits in this queue.
        let _ = self.queue.dispatch_pending(&mut self.state);
        self.state.pressed.take()
    }
}

impl Drop for WaylandTap {
    fn drop(&mut self) {
        if self.keyboard.version() >= 3 {
            self.keyboard.release();
        }
        if self.seat.version() >= 5 {
            self.seat.release();
        }
        let _ = self.connection.flush();
    }
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
        match event {
            wl_keyboard::Event::Keymap { format: WEnum::Value(wl_keyboard::KeymapFormat::XkbV1), fd, size } => {
                state.keymap = read_keymap(&fd, size as usize);
            }
            wl_keyboard::Event::Key { key, state: WEnum::Value(wl_keyboard::KeyState::Pressed), .. } => state.pressed = Some(key),
            wl_keyboard::Event::Modifiers { group, .. } => state.layout = group,
            _ => {}
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
