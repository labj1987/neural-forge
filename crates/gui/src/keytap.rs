//! The pressed key's hardware code and the session's keymap, for the toggle key row.
//!
//! The toolkit's key events name a key (by what it gives on the layout) but do not say which key it was, and the
//! layer reads keys from `/dev/input` by their place on the keyboard. So the display server is asked directly, over a
//! second line to it that is opened once for the window's life ([`KeyTap::open`]):
//!
//! - Wayland: a guest registry and seat on the toolkit's own connection. While the row waits for a key
//!   ([`KeyTap::listen`]) a second `wl_keyboard` receives the window's key presses with their evdev codes (a
//!   compositor sends them to every keyboard object of the focused client); the same kind of keyboard, held only for
//!   a moment, brings the keymap and the layout in use ([`KeyTap::keys`]). A keyboard is asked for only while the seat
//!   says it has one: a headless GNOME Shell 50 crashed in libmutter-mtk when asked for the keyboard of a seat without.
//! - X11: a private connection. While the row waits, XInput 2 raw key presses are selected on the root window (their
//!   detail is the keycode; version 2.1, so they also arrive while a window manager holds a passive grab); the layout's
//!   names are the root window's `_XKB_RULES_NAMES` and the layout in use is the XKB state's group.
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
use x11rb::protocol::xkb::{self, ConnectionExt as _};
use x11rb::protocol::xproto::{AtomEnum, ConnectionExt as _};
use x11rb::protocol::Event;
use x11rb::rust_connection::RustConnection;

use crate::keymap::SessionKeys;

type Failure = Box<dyn std::error::Error>;

/// A second line to the display server's keyboard, for the window's life.
pub enum KeyTap {
    Wayland(WaylandGuest),
    X11(Box<X11Tap>),
}

impl KeyTap {
    /// Opens the line on `window`'s display server. `None` where that cannot be done (the reason logged): the row
    /// then names keys by xkbcommon's default layout and goes by the key's name.
    pub fn open(window: &Window) -> Option<Self> {
        let display = HasDisplayHandle::display_handle(window).ok()?;
        let opened = match display.as_raw() {
            // SAFETY: the pointer is the live window's wl_display, this runs on the UI thread, which owns the
            // connection, and the tap lives in the window's view, so it is dropped while the display is open.
            RawDisplayHandle::Wayland(display) => unsafe { WaylandGuest::open(display.display.as_ptr()) }.map(KeyTap::Wayland),
            RawDisplayHandle::Xcb(_) | RawDisplayHandle::Xlib(_) => X11Tap::open().map(|tap| KeyTap::X11(Box::new(tap))),
            _ => return None,
        };
        opened.inspect_err(|e| log::info!("the toggle key goes by key names: no line to the display server's keyboard ({e})")).ok()
    }

    /// Starts hearing key presses (the row waits for a key). False where the display server will not report them.
    pub fn listen(&mut self) -> bool {
        let listening = match self {
            KeyTap::Wayland(guest) => guest.listen(),
            KeyTap::X11(tap) => tap.select(true),
        };
        listening.inspect_err(|e| log::info!("the toggle key goes by the key's name this time: no direct key events ({e})")).is_ok()
    }

    /// Stops hearing key presses.
    pub fn stop(&mut self) {
        match self {
            KeyTap::Wayland(guest) => guest.release_keyboard(),
            KeyTap::X11(tap) => {
                let _ = tap.select(false);
            }
        }
    }

    /// The evdev code of the key pressed last since [`KeyTap::listen`], if the display server reported one.
    pub fn last_press(&mut self) -> Option<u32> {
        match self {
            KeyTap::Wayland(guest) => guest.last_press(),
            KeyTap::X11(tap) => tap.last_press(),
        }
    }

    /// Whether the display server has said which layout is in use (an X server always does; a Wayland compositor may
    /// not until the modifiers change). Where it has not, the caller keeps what a key press told it
    /// (`SessionKeys::layout_of_press`).
    pub fn layout_known(&self) -> bool {
        match self {
            KeyTap::Wayland(guest) => guest.state.layout.is_some(),
            KeyTap::X11(tap) => tap.xkb,
        }
    }

    /// The session's keymap now, with the layout in use where the display server says (on Wayland it says so to the
    /// window that has the focus).
    pub fn keys(&mut self) -> Option<SessionKeys> {
        match self {
            KeyTap::Wayland(guest) => guest.keys(),
            KeyTap::X11(tap) => tap.keys(),
        }
        .filter(SessionKeys::known)
    }
}

// ── Wayland ──────────────────────────────────────────────────────────────────

#[derive(Default)]
struct GuestState {
    has_keyboard: bool,
    keymap: Option<String>,
    /// The layout in use (the keyboard's group), once a modifiers event has said (some compositors send one to a new
    /// keyboard of the focused window, others only when the modifiers next change).
    layout: Option<u32>,
    pressed: Option<u32>,
}

/// A guest registry and seat on the toolkit's own Wayland connection, and a keyboard of that seat while one is wanted.
pub struct WaylandGuest {
    queue: EventQueue<GuestState>,
    state: GuestState,
    connection: Connection,
    seat: WlSeat,
    keyboard: Option<WlKeyboard>,
}

impl WaylandGuest {
    /// # Safety
    /// `display` must be a live `wl_display`, no other thread may be reading the connection, and the guest must be
    /// dropped while the display is still open.
    unsafe fn open(display: *mut std::ffi::c_void) -> Result<Self, Failure> {
        // A guest handle on the toolkit's connection: dropping it closes nothing.
        let connection = Connection::from_backend(Backend::from_foreign_display(display.cast()));
        let (globals, mut queue) = registry_queue_init::<GuestState>(&connection)?;
        let seat: WlSeat = globals.bind(&queue.handle(), 1..=5, ())?;
        let mut state = GuestState::default();
        // The seat says what it has.
        queue.roundtrip(&mut state)?;
        Ok(Self { queue, state, connection, seat, keyboard: None })
    }

    /// Takes a keyboard of the seat (if it has none yet) and hears what comes with it: the keymap, and for the window
    /// that has the focus the layout in use.
    fn take_keyboard(&mut self) -> Result<(), Failure> {
        // Capabilities may have changed since the last look (a keyboard plugged in or out).
        self.queue.dispatch_pending(&mut self.state)?;
        if !self.state.has_keyboard {
            self.release_keyboard();
            return Err("the seat has no keyboard".into());
        }
        if self.keyboard.is_none() {
            // What an earlier keyboard heard of the layout in use is not this one's news: unknown until it says.
            self.state.layout = None;
            self.keyboard = Some(self.seat.get_keyboard(&self.queue.handle(), ()));
        }
        self.queue.roundtrip(&mut self.state)?;
        Ok(())
    }

    fn release_keyboard(&mut self) {
        if let Some(keyboard) = self.keyboard.take() {
            if keyboard.version() >= 3 {
                keyboard.release();
            }
            let _ = self.connection.flush();
        }
    }

    fn listen(&mut self) -> Result<(), Failure> {
        self.take_keyboard()?;
        self.state.pressed = None;
        Ok(())
    }

    fn last_press(&mut self) -> Option<u32> {
        // The toolkit has read the socket to deliver its own key event; this keyboard's copy waits in this queue.
        let _ = self.queue.dispatch_pending(&mut self.state);
        self.state.pressed.take()
    }

    fn keys(&mut self) -> Option<SessionKeys> {
        let listening = self.keyboard.is_some();
        let taken = self.take_keyboard();
        if !listening {
            self.release_keyboard();
        }
        taken.inspect_err(|e| log::info!("the toggle key is named by the default layout: the compositor's keymap could not be read ({e})")).ok()?;
        self.state.keymap.as_deref().map(|text| SessionKeys::from_text(text, self.state.layout.unwrap_or(0)))
    }
}

impl Drop for WaylandGuest {
    fn drop(&mut self) {
        self.release_keyboard();
        if self.seat.version() >= 5 {
            self.seat.release();
        }
        let _ = self.connection.flush();
    }
}

impl Dispatch<WlRegistry, GlobalListContents> for GuestState {
    fn event(_: &mut Self, _: &WlRegistry, _: <WlRegistry as Proxy>::Event, _: &GlobalListContents, _: &Connection, _: &QueueHandle<Self>) {}
}

impl Dispatch<WlSeat, ()> for GuestState {
    fn event(state: &mut Self, _: &WlSeat, event: wl_seat::Event, (): &(), _: &Connection, _: &QueueHandle<Self>) {
        if let wl_seat::Event::Capabilities { capabilities: WEnum::Value(capabilities) } = event {
            state.has_keyboard = capabilities.contains(wl_seat::Capability::Keyboard);
        }
    }
}

impl Dispatch<WlKeyboard, ()> for GuestState {
    fn event(state: &mut Self, _: &WlKeyboard, event: wl_keyboard::Event, (): &(), _: &Connection, _: &QueueHandle<Self>) {
        match event {
            wl_keyboard::Event::Keymap { format: WEnum::Value(wl_keyboard::KeymapFormat::XkbV1), fd, size } => {
                state.keymap = read_keymap(&fd, size as usize);
            }
            wl_keyboard::Event::Key { key, state: WEnum::Value(wl_keyboard::KeyState::Pressed), .. } => state.pressed = Some(key),
            wl_keyboard::Event::Modifiers { group, .. } => state.layout = Some(group),
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
    /// Whether the XKB extension answered (the layout in use is its state's group).
    xkb: bool,
}

/// How long a raw key press may take to arrive on this connection after the toolkit got the key on its own.
const RAW_KEY_WAIT: std::time::Duration = std::time::Duration::from_millis(60);

impl X11Tap {
    fn open() -> Result<Self, Failure> {
        let (connection, screen) = x11rb::connect(None)?;
        let root = connection.setup().roots.get(screen).ok_or("the X server names no such screen")?.root;
        // 2.1: raw events also go to this client while another (a window manager) holds a passive grab.
        connection.xinput_xi_query_version(2, 1)?.reply()?;
        let xkb = connection.xkb_use_extension(1, 0).ok().and_then(|cookie| cookie.reply().ok()).is_some_and(|reply| reply.supported);
        Ok(Self { connection, root, xkb })
    }

    /// Selects raw key presses on the root window, or nothing again.
    fn select(&mut self, on: bool) -> Result<(), Failure> {
        let events = if on { xinput::XIEventMask::RAW_KEY_PRESS } else { xinput::XIEventMask::from(0u32) };
        let mask = xinput::EventMask { deviceid: u16::from(xinput::Device::ALL_MASTER), mask: vec![events] };
        self.connection.xinput_xi_select_events(self.root, &[mask])?.check()?;
        // What was pressed before now is not the key waited for.
        while let Ok(Some(_)) = self.connection.poll_for_event() {}
        Ok(())
    }

    /// The layout the X server names (`_XKB_RULES_NAMES`: rules, model, layout, variant, options) and the one of its
    /// layouts in use (the XKB state's group).
    fn keys(&self) -> Option<SessionKeys> {
        let atom = self.connection.intern_atom(true, b"_XKB_RULES_NAMES").ok()?.reply().ok()?.atom;
        let property = self.connection.get_property(false, self.root, atom, AtomEnum::STRING, 0, 1024).ok()?.reply().ok()?;
        let names: Vec<String> = property.value.split(|&b| b == 0).map(|name| String::from_utf8_lossy(name).into_owned()).collect();
        let name = |i: usize| names.get(i).map_or("", String::as_str);
        if name(2).is_empty() {
            return None;
        }
        let group = self.xkb.then(|| self.connection.xkb_get_state(u16::from(xkb::ID::USE_CORE_KBD)).ok()?.reply().ok()).flatten();
        let layout = group.map_or(0, |state| u32::from(u8::from(state.group)));
        Some(SessionKeys::from_names(name(0), name(1), name(2), name(3), name(4)).in_layout(layout))
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
