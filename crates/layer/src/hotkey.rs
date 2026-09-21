//! An in-game key, in every environment the layer runs in.
//!
//! The layer has no window: it is a shared object inside someone else's process, and the
//! environments it has to work in (X11/XWayland, a Wayland-native Wine, gamescope) do not
//! agree on what "the keyboard" is. Two backends, tried in this order:
//!
//! * **evdev** reads the kernel's input devices directly. It works everywhere, since a Proton
//!   game is still a Linux process, and has the lowest latency, but keyboards are `root:input`
//!   with no uaccess ACL, so it needs the `input` group.
//! * **XInput2** raw key events selected on the root window. It needs no permission and raw
//!   events arrive whichever client has focus (including when the focused client is Wayland).
//!   `XQueryKeymap`, which this layer used before, reports nothing for a real key press under
//!   XWayland, so it is not an option.
//!
//! Neither links X11: libX11/libXi are `dlopen`ed only when the evdev backend is unavailable, so
//! a Vulkan process on a machine with no X at all is unaffected.
//!
//! Keys are Linux `KEY_*` codes throughout (what the GUI persists). Env overrides:
//! `NEURAL_FORGE_TOGGLE_KEY` (a key name such as `F10`, or a bare number) replaces the key the
//! header names, and `NEURAL_FORGE_HOTKEY_BACKEND` (`evdev` or `x11`) forces one backend.

use std::collections::{HashMap, HashSet};
use std::ffi::{c_char, c_int, c_long, c_uint, c_ulong, c_void, CString};
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// The keys anyone would plausibly bind. Not the whole table: a name that is not here simply
/// does not resolve, which is better than accepting something nobody can type.
const KEYS: &[(&str, u32)] = &[
    ("F1", 59), ("F2", 60), ("F3", 61), ("F4", 62), ("F5", 63), ("F6", 64),
    ("F7", 65), ("F8", 66), ("F9", 67), ("F10", 68), ("F11", 87), ("F12", 88),
    ("HOME", 102), ("END", 107), ("INSERT", 110), ("DELETE", 111),
    ("PAGEUP", 104), ("PAGEDOWN", 109), ("PAUSE", 119), ("SCROLLLOCK", 70),
    ("SYSRQ", 99), ("GRAVE", 41),
    ("A", 30), ("B", 48), ("C", 46), ("D", 32), ("E", 18), ("F", 33), ("G", 34),
    ("H", 35), ("I", 23), ("J", 36), ("K", 37), ("L", 38), ("M", 50), ("N", 49),
    ("O", 24), ("P", 25), ("Q", 16), ("R", 19), ("S", 31), ("T", 20), ("U", 22),
    ("V", 47), ("W", 17), ("X", 45), ("Y", 21), ("Z", 44),
    ("0", 11), ("1", 2), ("2", 3), ("3", 4), ("4", 5), ("5", 6), ("6", 7), ("7", 8), ("8", 9), ("9", 10),
];

/// A Linux key code from a name (`F10`, `Home`, `KEY_N`, case-insensitive) or a bare number, or 0.
pub fn key_code_from_name(name: &str) -> u32 {
    let name = name.trim();
    if name.is_empty() {
        return 0;
    }
    if name.bytes().all(|b| b.is_ascii_digit()) {
        // A bare number is a Linux key code, so a key with no name here is still reachable.
        return name.parse().unwrap_or(0);
    }
    let upper = name.to_ascii_uppercase();
    let upper = upper.strip_prefix("KEY_").unwrap_or(&upper);
    KEYS.iter().find(|(n, _)| *n == upper).map_or(0, |&(_, code)| code)
}

/// `NEURAL_FORGE_TOGGLE_KEY`, if set to something that resolves.
fn env_toggle_key() -> Option<u32> {
    let code = key_code_from_name(&neural_forge_protocol::env::var("NEURAL_FORGE_TOGGLE_KEY")?);
    (code != 0).then_some(code)
}

/// The key to watch: the environment's override, else the header's configured key.
pub fn effective_key(configured: u32) -> u32 {
    env_toggle_key().unwrap_or(configured)
}

// ---- evdev ----------------------------------------------------------------------------------

const EV_KEY: u16 = 0x01;
const EV_MAX: usize = 0x1f;
const KEY_MAX: usize = 0x2ff;
const KEY_A: usize = 30;
const KEY_Z: usize = 44;

const fn ioc_read(nr: u32, size: usize) -> c_ulong {
    // _IOC(_IOC_READ, 'E', nr, size)
    ((2u64 << 30) | ((size as u64) << 16) | (0x45u64 << 8) | nr as u64) as c_ulong
}
const EVIOCGVERSION: c_ulong = ioc_read(0x01, std::mem::size_of::<c_int>());
const fn eviocgbit(ev: u32, len: usize) -> c_ulong {
    ioc_read(0x20 + ev, len)
}

const LONG_BITS: usize = 8 * std::mem::size_of::<c_ulong>();

fn test_bit(bits: &[c_ulong], bit: usize) -> bool {
    (bits[bit / LONG_BITS] >> (bit % LONG_BITS)) & 1 == 1
}

/// A keyboard, as opposed to a mouse, lid switch, power button or gamepad: nothing else on a
/// normal system claims every key from A to Z.
fn looks_like_a_keyboard(fd: c_int) -> bool {
    let mut evbits = [0 as c_ulong; (EV_MAX + LONG_BITS) / LONG_BITS];
    let mut keybits = [0 as c_ulong; (KEY_MAX + LONG_BITS) / LONG_BITS];
    // SAFETY: each buffer is exactly the length the request encodes.
    unsafe {
        if libc::ioctl(fd, eviocgbit(0, std::mem::size_of_val(&evbits)), evbits.as_mut_ptr()) < 0 {
            return false;
        }
        if !test_bit(&evbits, EV_KEY as usize) {
            return false;
        }
        if libc::ioctl(fd, eviocgbit(EV_KEY as u32, std::mem::size_of_val(&keybits)), keybits.as_mut_ptr()) < 0 {
            return false;
        }
    }
    (KEY_A..=KEY_Z).all(|k| test_bit(&keybits, k))
}

struct Evdev {
    /// One open, non-blocking descriptor per keyboard, keyed by its `event*` name.
    fds: HashMap<String, c_int>,
    /// Nodes judged not to be keyboards, by name and inode. Remembering the rejection is what
    /// keeps a rescan cheap (opening and closing an evdev node costs milliseconds on the present
    /// thread); the inode is what stops a replugged device that reuses the name inheriting the
    /// old verdict.
    not_keyboard: HashMap<String, u64>,
    last_scan: Option<Instant>,
    announced: bool,
}

impl Evdev {
    fn open() -> Option<Self> {
        let mut e = Evdev { fds: HashMap::new(), not_keyboard: HashMap::new(), last_scan: None, announced: false };
        e.rescan();
        (!e.fds.is_empty()).then_some(e)
    }

    /// Opens any keyboard not already open and drops any that went away.
    fn rescan(&mut self) {
        // Gone devices are detected with EVIOCGVERSION, not a read: a read answers with a queued
        // event first, so a dead device with unread input would look alive and its name would
        // stay taken, and the device that next got that name would never be opened.
        self.fds.retain(|_, &mut fd| {
            let mut version: c_int = 0;
            // SAFETY: `fd` is one of our open descriptors; `version` is a valid c_int.
            let alive = unsafe { libc::ioctl(fd, EVIOCGVERSION, &mut version as *mut c_int) } >= 0;
            if !alive {
                // SAFETY: closed exactly once, right here.
                unsafe { libc::close(fd) };
            }
            alive
        });
        let Ok(dir) = std::fs::read_dir("/dev/input") else { return };
        let mut seen = HashSet::new();
        for entry in dir.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if !name.starts_with("event") {
                continue;
            }
            seen.insert(name.clone());
            if self.fds.contains_key(&name) {
                continue;
            }
            let path: PathBuf = entry.path();
            let inode = {
                use std::os::unix::fs::MetadataExt;
                std::fs::metadata(&path).ok().map(|m| m.ino())
            };
            if let (Some(ino), Some(&judged)) = (inode, self.not_keyboard.get(&name)) {
                if ino == judged {
                    continue;
                }
                self.not_keyboard.remove(&name); // same path, different device: look again
            }
            let Ok(c_path) = CString::new(path.as_os_str().as_encoded_bytes()) else { continue };
            // SAFETY: `c_path` is a valid NUL-terminated path.
            let fd = unsafe { libc::open(c_path.as_ptr(), libc::O_RDONLY | libc::O_NONBLOCK | libc::O_CLOEXEC) };
            if fd < 0 {
                continue;
            }
            if !looks_like_a_keyboard(fd) {
                // SAFETY: `fd` was just opened above.
                unsafe { libc::close(fd) };
                if let Some(ino) = inode {
                    self.not_keyboard.insert(name, ino);
                }
                continue;
            }
            if self.announced {
                crate::log!("[hotkey] picked up a keyboard that appeared later: {}", path.display());
            }
            self.fds.insert(name, fd);
        }
        self.not_keyboard.retain(|name, _| seen.contains(name));
        self.announced = true;
    }

    /// Drains every keyboard and appends the codes that went down. Drains rather than looking
    /// for one code: a read that leaves events behind makes the next answer stale.
    fn poll(&mut self, pending: &mut Vec<u32>) {
        let now = Instant::now();
        if self.last_scan.is_none_or(|t| now.duration_since(t) >= Duration::from_secs(1)) {
            self.last_scan = Some(now);
            self.rescan();
        }
        // SAFETY: an all-zero `input_event` is a valid value.
        let mut events: [libc::input_event; 64] = unsafe { std::mem::zeroed() };
        let size = std::mem::size_of_val(&events);
        for &fd in self.fds.values() {
            loop {
                // SAFETY: `events` is `size` writable bytes.
                let n = unsafe { libc::read(fd, events.as_mut_ptr().cast::<c_void>(), size) };
                if n <= 0 {
                    break;
                }
                let count = n as usize / std::mem::size_of::<libc::input_event>();
                pending.extend(events[..count].iter().filter(|e| e.type_ == EV_KEY && e.value == 1).map(|e| u32::from(e.code)));
                if (n as usize) < size {
                    break;
                }
            }
        }
    }
}

impl Drop for Evdev {
    fn drop(&mut self) {
        for &fd in self.fds.values() {
            // SAFETY: each descriptor is ours and closed exactly once.
            unsafe { libc::close(fd) };
        }
    }
}

// ---- XInput2 --------------------------------------------------------------------------------

const GENERIC_EVENT: c_int = 35;
const XI_RAW_KEY_PRESS: c_int = 13;
const XI_ALL_MASTER_DEVICES: c_int = 1;

#[repr(C)]
struct XGenericEventCookie {
    type_: c_int,
    serial: c_ulong,
    send_event: c_int,
    display: *mut c_void,
    extension: c_int,
    evtype: c_int,
    cookie: c_uint,
    data: *mut c_void,
}

/// The leading fields of `XIRawEvent`, up to the key's `detail`.
#[repr(C)]
struct XiRawEventHead {
    type_: c_int,
    serial: c_ulong,
    send_event: c_int,
    display: *mut c_void,
    extension: c_int,
    evtype: c_int,
    time: c_ulong,
    deviceid: c_int,
    sourceid: c_int,
    detail: c_int,
}

#[repr(C)]
struct XiEventMask {
    deviceid: c_int,
    mask_len: c_int,
    mask: *mut u8,
}

/// `XEvent` is a union whose largest member is 24 longs.
#[repr(C)]
struct XEvent {
    pad: [c_long; 24],
}

struct X11 {
    x11: *mut c_void,
    xi: *mut c_void,
    display: *mut c_void,
    opcode: c_int,
    close_display: unsafe extern "C" fn(*mut c_void) -> c_int,
    pending: unsafe extern "C" fn(*mut c_void) -> c_int,
    next_event: unsafe extern "C" fn(*mut c_void, *mut XEvent) -> c_int,
    get_event_data: unsafe extern "C" fn(*mut c_void, *mut XGenericEventCookie) -> c_int,
    free_event_data: unsafe extern "C" fn(*mut c_void, *mut XGenericEventCookie),
}

/// Resolves `name` in `lib` as a function pointer of type `F`.
///
/// # Safety
/// `F` must be the function-pointer type matching `name`'s real signature.
unsafe fn sym<F: Copy>(lib: *mut c_void, name: &str) -> Option<F> {
    let c = CString::new(name).ok()?;
    // SAFETY: `lib` is a live handle from dlopen; `c` is NUL-terminated.
    let p = unsafe { libc::dlsym(lib, c.as_ptr()) };
    // SAFETY: forwarded from this function's contract.
    (!p.is_null()).then(|| unsafe { std::mem::transmute_copy::<*mut c_void, F>(&p) })
}

impl X11 {
    fn open() -> Option<Self> {
        std::env::var_os("DISPLAY")?;
        // Loaded, not linked, so a layer on a machine with no X at all still starts.
        // SAFETY: plain dlopen calls with NUL-terminated names.
        let (x11, xi) = unsafe {
            (
                libc::dlopen(c"libX11.so.6".as_ptr(), libc::RTLD_LAZY | libc::RTLD_LOCAL),
                libc::dlopen(c"libXi.so.6".as_ptr(), libc::RTLD_LAZY | libc::RTLD_LOCAL),
            )
        };
        let close_libs = |x11: *mut c_void, xi: *mut c_void| unsafe {
            if !xi.is_null() {
                libc::dlclose(xi);
            }
            if !x11.is_null() {
                libc::dlclose(x11);
            }
        };
        if x11.is_null() || xi.is_null() {
            close_libs(x11, xi);
            return None;
        }
        // SAFETY: every type below matches the named Xlib/XInput2 function's C signature.
        let resolved = unsafe {
            (|| {
                let open_display: unsafe extern "C" fn(*const c_char) -> *mut c_void = sym(x11, "XOpenDisplay")?;
                let query_extension: unsafe extern "C" fn(*mut c_void, *const c_char, *mut c_int, *mut c_int, *mut c_int) -> c_int =
                    sym(x11, "XQueryExtension")?;
                let default_root: unsafe extern "C" fn(*mut c_void) -> c_ulong = sym(x11, "XDefaultRootWindow")?;
                let flush: unsafe extern "C" fn(*mut c_void) -> c_int = sym(x11, "XFlush")?;
                let query_version: unsafe extern "C" fn(*mut c_void, *mut c_int, *mut c_int) -> c_int = sym(xi, "XIQueryVersion")?;
                let select_events: unsafe extern "C" fn(*mut c_void, c_ulong, *mut XiEventMask, c_int) -> c_int =
                    sym(xi, "XISelectEvents")?;
                let close_display = sym(x11, "XCloseDisplay")?;
                let pending = sym(x11, "XPending")?;
                let next_event = sym(x11, "XNextEvent")?;
                let get_event_data = sym(x11, "XGetEventData")?;
                let free_event_data = sym(x11, "XFreeEventData")?;

                let display = open_display(std::ptr::null());
                if display.is_null() {
                    return None;
                }
                let fail = |display| {
                    // The connection was opened above; give it back on every failure path.
                    let close: unsafe extern "C" fn(*mut c_void) -> c_int = close_display;
                    close(display);
                    None::<()>
                };
                let (mut opcode, mut event, mut error) = (0, 0, 0);
                if query_extension(display, c"XInputExtension".as_ptr(), &mut opcode, &mut event, &mut error) == 0 {
                    fail(display);
                    return None;
                }
                let (mut major, mut minor) = (2, 2);
                if query_version(display, &mut major, &mut minor) != 0 {
                    fail(display);
                    return None;
                }
                // Raw key presses on the root window: delivered whatever has focus, which is the
                // point — the layer has no window and the game may not even be an X client.
                let mut mask = [0u8; 4];
                mask[(XI_RAW_KEY_PRESS / 8) as usize] |= 1 << (XI_RAW_KEY_PRESS % 8);
                let mut em = XiEventMask { deviceid: XI_ALL_MASTER_DEVICES, mask_len: mask.len() as c_int, mask: mask.as_mut_ptr() };
                select_events(display, default_root(display), &mut em, 1);
                flush(display);
                Some((display, opcode, close_display, pending, next_event, get_event_data, free_event_data))
            })()
        };
        let Some((display, opcode, close_display, pending, next_event, get_event_data, free_event_data)) = resolved else {
            close_libs(x11, xi);
            return None;
        };
        Some(X11 { x11, xi, display, opcode, close_display, pending, next_event, get_event_data, free_event_data })
    }

    /// Raw key presses are already edges, so no held state is tracked. An X keycode is the evdev
    /// code plus eight on every evdev-driven X server, which is all of them on Linux.
    fn poll(&mut self, pending: &mut Vec<u32>) {
        // SAFETY: `display` is live until Drop; the event/cookie structs match Xlib's layout for
        // the fields read, and every cookie fetched is freed.
        unsafe {
            while (self.pending)(self.display) > 0 {
                let mut event = XEvent { pad: [0; 24] };
                (self.next_event)(self.display, &mut event);
                let cookie = (&mut event as *mut XEvent).cast::<XGenericEventCookie>();
                if (*cookie).type_ != GENERIC_EVENT || (*cookie).extension != self.opcode {
                    continue;
                }
                if (self.get_event_data)(self.display, cookie) == 0 {
                    continue;
                }
                if (*cookie).evtype == XI_RAW_KEY_PRESS {
                    let raw = (*cookie).data.cast::<XiRawEventHead>();
                    if (*raw).detail >= 8 {
                        pending.push((*raw).detail as u32 - 8);
                    }
                }
                (self.free_event_data)(self.display, cookie);
            }
        }
    }
}

impl Drop for X11 {
    fn drop(&mut self) {
        // SAFETY: the connection is closed exactly once, before the libraries that own the
        // code doing so are unloaded.
        unsafe {
            (self.close_display)(self.display);
            libc::dlclose(self.xi);
            libc::dlclose(self.x11);
        }
    }
}

// ---- the poller -----------------------------------------------------------------------------

enum Backend {
    Unopened,
    Evdev(Evdev),
    X11(X11),
    None,
}

pub struct Poller {
    backend: Backend,
    /// Codes seen down since the last ask, so one drain serves any key the caller asks about.
    pending: Vec<u32>,
    last_poll: Option<Instant>,
}

// The backends hold raw descriptors/pointers used only from the layer's serialized present hook.
unsafe impl Send for Poller {}

impl Default for Poller {
    fn default() -> Self {
        Self { backend: Backend::Unopened, pending: Vec::new(), last_poll: None }
    }
}

/// Reading the keyboards is cheap, but not free on every frame at every frame rate. Short enough
/// that a normal key tap (well over 50 ms) is always seen.
const POLL_INTERVAL: Duration = Duration::from_millis(20);

impl Poller {
    fn open(&mut self) {
        let forced = neural_forge_protocol::env::var("NEURAL_FORGE_HOTKEY_BACKEND");
        let forced = forced.as_deref();
        let (want_evdev, want_x11) = (forced.is_none_or(|f| f == "evdev"), forced.is_none_or(|f| f == "x11"));
        if want_evdev {
            if let Some(e) = Evdev::open() {
                crate::log!("[hotkey] watching {} keyboard(s) through evdev", e.fds.len());
                self.backend = Backend::Evdev(e);
                return;
            }
        }
        if want_x11 {
            if let Some(x) = X11::open() {
                crate::log!("[hotkey] watching XInput2 raw keys on {}", std::env::var("DISPLAY").unwrap_or_default());
                self.backend = Backend::X11(x);
                return;
            }
        }
        crate::log!(
            "[hotkey] no way to read the keyboard here: /dev/input needs the 'input' group or a uaccess ACL, \
             and without it only an X11 session can be read (never one inside gamescope)"
        );
        self.backend = Backend::None;
    }

    /// Returns true once, on the press of `evdev_code` (`NEURAL_FORGE_TOGGLE_KEY` replaces it
    /// when set).
    pub fn pressed(&mut self, evdev_code: u32) -> bool {
        let key = effective_key(evdev_code);
        if key == 0 {
            return false;
        }
        if matches!(self.backend, Backend::Unopened) {
            self.open();
        }
        let now = Instant::now();
        // Drained events stay in `pending`, so throttling delays a press but never loses it.
        if self.last_poll.is_none_or(|t| now.duration_since(t) >= POLL_INTERVAL) {
            self.last_poll = Some(now);
            match &mut self.backend {
                Backend::Evdev(e) => e.poll(&mut self.pending),
                Backend::X11(x) => x.poll(&mut self.pending),
                Backend::Unopened | Backend::None => {}
            }
        }
        if let Some(i) = self.pending.iter().position(|&c| c == key) {
            self.pending.remove(i);
            true
        } else {
            // Nothing this caller wanted; keep the list from growing forever.
            if self.pending.len() > 256 {
                self.pending.clear();
            }
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_resolve_case_insensitively_with_or_without_the_prefix() {
        assert_eq!(key_code_from_name("F11"), 87);
        assert_eq!(key_code_from_name("f10"), 68);
        assert_eq!(key_code_from_name("KEY_HOME"), 102);
        assert_eq!(key_code_from_name("n"), 49);
        assert_eq!(key_code_from_name(""), 0);
        assert_eq!(key_code_from_name("nonsense"), 0);
    }

    #[test]
    fn bare_numbers_are_key_codes() {
        assert_eq!(key_code_from_name("87"), 87);
        assert_eq!(key_code_from_name("183"), 183);
        assert_eq!(key_code_from_name("5"), 5, "a bare number is a code, not a digit key's name");
    }

    #[test]
    fn evdev_ioctl_numbers_match_the_kernel_macros() {
        // Values from <linux/input.h> on x86-64: EVIOCGVERSION and EVIOCGBIT(EV_KEY, 96).
        assert_eq!(EVIOCGVERSION as u64, 0x8004_4501);
        assert_eq!(eviocgbit(1, 96) as u64, 0x8060_4521);
    }

    #[test]
    fn a_press_seen_by_one_drain_is_delivered_once() {
        let mut p = Poller::default();
        p.backend = Backend::None;
        p.pending.push(87);
        p.last_poll = Some(Instant::now());
        assert!(p.pressed(87));
        assert!(!p.pressed(87));
    }
}
