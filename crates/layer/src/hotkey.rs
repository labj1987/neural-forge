//! An in-game key, in every environment the layer runs in.
//!
//! The layer has no window: it is a shared object inside someone else's process, and the
//! environments it has to work in (X11/XWayland, a Wayland-native Wine, gamescope) do not
//! agree on what "the keyboard" is. Two backends, tried in this order:
//!
//! * **evdev** reads the kernel's input devices directly. It works everywhere, since a Proton
//!   game is still a Linux process, and has the lowest latency, but keyboards are `root:input`
//!   with no uaccess ACL, so it needs the `input` group.
//! * **XInput2** raw key events selected on the root window. It needs no permission. On an X
//!   server raw events arrive whichever client has focus; under XWayland only while an X client
//!   (a Proton game is one) has the keyboard focus, so a press in a native Wayland window is never
//!   seen. `XQueryKeymap`, which this layer used before, reports nothing for a real key press under
//!   XWayland, so it is not an option. The connection is the layer's own: an I/O error on it (the
//!   X server going away) marks the backend dead instead of letting Xlib end the game's process.
//!
//! Both backends see presses whatever has the focus, so a press acts only while the game has it:
//! the window the X server (XWayland) names active (`_NET_ACTIVE_WINDOW`) belongs to this process
//! ([`Focus`]). Where that cannot be told (no X server, or one without EWMH, as inside gamescope,
//! whose nested server holds only the game) a press acts; `NEURAL_FORGE_HOTKEY_GLOBAL=1` makes it act
//! even while another client is known to have the focus. Only presses of the watched key are kept;
//! every other key code read is dropped at once.
//!
//! Neither links X11: libX11/libXi are `dlopen`ed only when the evdev backend is unavailable, so
//! a Vulkan process on a machine with no X at all is unaffected.
//!
//! Keys are Linux `KEY_*` codes throughout (what the GUI persists). Env overrides:
//! `NEURAL_FORGE_TOGGLE_KEY` (a key name such as `F10`, or a bare number) replaces the key the
//! header names, and `NEURAL_FORGE_HOTKEY_BACKEND` (`evdev` or `x11`) forces one backend.

use std::collections::{HashMap, HashSet};
use std::ffi::{c_char, c_int, c_long, c_uint, c_ulong, c_void, CStr, CString};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering};
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
    /// Presses of `key` since the last poll.
    fn poll(&mut self, key: u32) -> u32 {
        let mut presses = 0;
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
                presses += events[..count].iter().filter(|e| e.type_ == EV_KEY && e.value == 1 && u32::from(e.code) == key).count() as u32;
                if (n as usize) < size {
                    break;
                }
            }
        }
        presses
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
    /// This connection's entry in [`X11_DISPLAYS`] and [`X11_DEAD`].
    slot: usize,
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
                // Needed to survive the connection's death (libX11 1.7 and later): without it, Xlib's
                // I/O error path ends the process, so the backend is not used at all.
                let set_io_error_handler: SetIoErrorHandler = sym(x11, "XSetIOErrorHandler")?;
                let set_io_error_exit_handler: unsafe extern "C" fn(*mut c_void, Option<IoErrorExitHandler>, *mut c_void) =
                    sym(x11, "XSetIOErrorExitHandler")?;
                let pending = sym(x11, "XPending")?;
                let next_event = sym(x11, "XNextEvent")?;
                let get_event_data = sym(x11, "XGetEventData")?;
                let free_event_data = sym(x11, "XFreeEventData")?;

                let display = open_display(std::ptr::null());
                if display.is_null() {
                    return None;
                }
                let fail = |display, slot: Option<usize>| {
                    // The connection was opened above; give it back on every failure path, then its
                    // entry (after the close, so an error the close itself hits is still ours).
                    let close: unsafe extern "C" fn(*mut c_void) -> c_int = close_display;
                    close(display);
                    if let Some(slot) = slot {
                        release_display(slot);
                    }
                    None::<()>
                };
                // The I/O error handler is process-wide (the game's own Xlib use, Wine's, shares it):
                // ours acts on the layer's own connections only and hands every other one to the
                // handler that was there. It is installed once per process. The exit handler is this
                // connection's own; returning from it keeps the process alive.
                let Some(slot) = register_display(display) else {
                    fail(display, None);
                    return None;
                };
                if !install_io_error_handler(set_io_error_handler) {
                    fail(display, Some(slot));
                    return None;
                }
                set_io_error_exit_handler(display, Some(io_error_exit), slot as *mut c_void);
                let (mut opcode, mut event, mut error) = (0, 0, 0);
                if query_extension(display, c"XInputExtension".as_ptr(), &mut opcode, &mut event, &mut error) == 0 {
                    fail(display, Some(slot));
                    return None;
                }
                let (mut major, mut minor) = (2, 2);
                if query_version(display, &mut major, &mut minor) != 0 {
                    fail(display, Some(slot));
                    return None;
                }
                // Raw key presses on the root window: delivered whatever has focus, which is the
                // point — the layer has no window and the game may not even be an X client.
                let mut mask = [0u8; 4];
                mask[(XI_RAW_KEY_PRESS / 8) as usize] |= 1 << (XI_RAW_KEY_PRESS % 8);
                let mut em = XiEventMask { deviceid: XI_ALL_MASTER_DEVICES, mask_len: mask.len() as c_int, mask: mask.as_mut_ptr() };
                select_events(display, default_root(display), &mut em, 1);
                flush(display);
                Some((display, slot, opcode, close_display, pending, next_event, get_event_data, free_event_data))
            })()
        };
        let Some((display, slot, opcode, close_display, pending, next_event, get_event_data, free_event_data)) = resolved else {
            close_libs(x11, xi);
            return None;
        };
        Some(X11 { x11, xi, display, slot, opcode, close_display, pending, next_event, get_event_data, free_event_data })
    }

    /// Raw key presses are already edges, so no held state is tracked. An X keycode is the evdev
    /// code plus eight on every evdev-driven X server, which is all of them on Linux.
    /// Presses of `key` since the last poll; none once the connection died.
    fn poll(&mut self, key: u32) -> u32 {
        let mut presses = 0;
        let dead = &X11_DEAD[self.slot];
        if dead.load(Ordering::Acquire) {
            return 0;
        }
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
                    if (*raw).detail >= 8 && (*raw).detail as u32 - 8 == key {
                        presses += 1;
                    }
                }
                (self.free_event_data)(self.display, cookie);
                if dead.load(Ordering::Acquire) {
                    break;
                }
            }
        }
        presses
    }
}

impl Drop for X11 {
    fn drop(&mut self) {
        // SAFETY: the connection is closed exactly once, before the libraries that own the
        // code doing so are unloaded.
        // A connection Xlib reported an I/O error on is unusable: it (and the libraries its code is
        // in, which the error path may still reference) is left as it is, and keeps its entry so a
        // late error on it is still recognised as ours.
        if X11_DEAD[self.slot].load(Ordering::Acquire) {
            return;
        }
        unsafe {
            (self.close_display)(self.display);
        }
        // After the close: nothing names the closed connection as the layer's any more, so a new
        // one Xlib hands out at the same address is not mistaken for it.
        release_display(self.slot);
        unsafe {
            libc::dlclose(self.xi);
            libc::dlclose(self.x11);
        }
    }
}

type IoErrorHandler = unsafe extern "C" fn(*mut c_void) -> c_int;
type IoErrorExitHandler = unsafe extern "C" fn(*mut c_void, *mut c_void);
type SetIoErrorHandler = unsafe extern "C" fn(Option<IoErrorHandler>) -> Option<IoErrorHandler>;

/// How many of the layer's own X connections can be open at once: one per device's poller.
const X11_SLOTS: usize = 4;
/// The layer's own X connections (null: a free entry), whether Xlib reported an I/O error on each,
/// and the handler that was installed before ours. An entry whose connection died stays taken.
static X11_DISPLAYS: [AtomicPtr<c_void>; X11_SLOTS] = [const { AtomicPtr::new(std::ptr::null_mut()) }; X11_SLOTS];
static X11_DEAD: [AtomicBool; X11_SLOTS] = [const { AtomicBool::new(false) }; X11_SLOTS];
static PREVIOUS_IO_ERROR: AtomicUsize = AtomicUsize::new(0);

/// Records `display` as one of the layer's own connections: its entry, `None` when all are taken.
fn register_display(display: *mut c_void) -> Option<usize> {
    X11_DISPLAYS.iter().position(|entry| entry.compare_exchange(std::ptr::null_mut(), display, Ordering::AcqRel, Ordering::Acquire).is_ok())
}

/// Frees the entry of a connection that was closed without an I/O error.
fn release_display(slot: usize) {
    X11_DEAD[slot].store(false, Ordering::Release);
    X11_DISPLAYS[slot].store(std::ptr::null_mut(), Ordering::Release);
}

/// Installs [`io_error`] as Xlib's I/O error handler, once per process. The handler is
/// process-wide: a second `XSetIOErrorHandler` would hand back our own handler as the one before,
/// and [`io_error`] would then forward every other connection's error to itself until the stack
/// overflowed. The code the handler lives in is pinned first (`RTLD_NODELETE` on libX11 and on the
/// layer's own object): the Vulkan loader unloads a layer with the last instance, and Xlib would
/// then call into unmapped memory. `false` when they cannot be pinned (the X11 backend is then not
/// used, since Xlib's default handler ends the process on the connection's death).
fn install_io_error_handler(set: SetIoErrorHandler) -> bool {
    static INSTALLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *INSTALLED.get_or_init(|| {
        if !pin_handler_code() {
            crate::log!("[hotkey] could not pin the X I/O error handler's code; not using XInput2");
            return false;
        }
        // SAFETY: `set` is libX11's `XSetIOErrorHandler`; `io_error` has the handler's signature.
        chain_previous(unsafe { set(Some(io_error)) });
        true
    })
}

/// Keeps `previous` as the handler other connections' errors go to. Never our own: that would
/// forward to itself forever.
fn chain_previous(previous: Option<IoErrorHandler>) {
    let previous = previous.map_or(0, |f| f as usize);
    if previous != io_error as IoErrorHandler as usize {
        PREVIOUS_IO_ERROR.store(previous, Ordering::Release);
    }
}

/// Makes libX11 and the layer's own object stay loaded for the life of the process.
fn pin_handler_code() -> bool {
    // SAFETY: `RTLD_NOLOAD` only looks up an object already loaded; the reference it takes is kept
    // on purpose, and `RTLD_NODELETE` keeps the object mapped past every `dlclose`.
    let pin = |name: &CStr| unsafe { !libc::dlopen(name.as_ptr(), libc::RTLD_LAZY | libc::RTLD_NOLOAD | libc::RTLD_NODELETE).is_null() };
    let Some(layer) = crate::layer_object_path().and_then(|p| CString::new(p).ok()) else { return false };
    pin(c"libX11.so.6") && pin(&layer)
}

/// Xlib's I/O error handler: one of the layer's own connections is marked dead (Xlib then calls
/// its exit handler, [`io_error_exit`], which returns); any other connection goes to the handler
/// installed before.
unsafe extern "C" fn io_error(display: *mut c_void) -> c_int {
    if !display.is_null() {
        if let Some(slot) = X11_DISPLAYS.iter().position(|entry| entry.load(Ordering::Acquire) == display) {
            X11_DEAD[slot].store(true, Ordering::Release);
            return 0;
        }
    }
    match PREVIOUS_IO_ERROR.load(Ordering::Acquire) {
        0 => 0,
        // SAFETY: the value is the handler `XSetIOErrorHandler` returned, with this signature.
        previous => unsafe { std::mem::transmute::<usize, IoErrorHandler>(previous)(display) },
    }
}

/// The exit handler of one of the layer's connections; `data` is its entry.
unsafe extern "C" fn io_error_exit(_display: *mut c_void, data: *mut c_void) {
    if let Some(dead) = X11_DEAD.get(data as usize) {
        dead.store(true, Ordering::Release);
    }
}

// ---- whether the game has the keyboard focus ----------------------------------------------------

/// The game's keyboard focus as the X server (or XWayland) sees it: the window EWMH names active
/// (`_NET_ACTIVE_WINDOW` on the root) belongs to this process (`_NET_WM_PID`, which Wine sets on its
/// windows). Through libxcb, `dlopen`ed: its requests return errors as values, so a query racing a
/// window's destruction, or the server going away, can never reach an Xlib error handler (which in
/// a game process may end it). Asked only when the toggle key was pressed.
struct Focus {
    lib: *mut c_void,
    conn: *mut c_void,
    root: u32,
    active: u32,
    pid: u32,
    client_list: u32,
    has_error: unsafe extern "C" fn(*mut c_void) -> c_int,
    get_property: unsafe extern "C" fn(*mut c_void, u8, u32, u32, u32, u32, u32) -> u32,
    get_property_reply: unsafe extern "C" fn(*mut c_void, u32, *mut *mut c_void) -> *mut c_void,
    value: unsafe extern "C" fn(*const c_void) -> *mut c_void,
    value_length: unsafe extern "C" fn(*const c_void) -> c_int,
    disconnect: unsafe extern "C" fn(*mut c_void),
}

/// `xcb_screen_iterator_t`, of which only `data` (whose first field is the root window) is read.
#[repr(C)]
struct XcbScreenIterator {
    data: *const u32,
    rem: c_int,
    index: c_int,
}

const XCB_ATOM_WINDOW: u32 = 33;
const XCB_ATOM_CARDINAL: u32 = 6;

impl Focus {
    fn open() -> Option<Self> {
        std::env::var_os("DISPLAY")?;
        // SAFETY: a plain dlopen of a NUL-terminated name.
        let lib = unsafe { libc::dlopen(c"libxcb.so.1".as_ptr(), libc::RTLD_LAZY | libc::RTLD_LOCAL) };
        if lib.is_null() {
            return None;
        }
        // SAFETY: every type below matches the named libxcb function's C signature.
        let focus = unsafe {
            (|| {
                let connect: unsafe extern "C" fn(*const c_char, *mut c_int) -> *mut c_void = sym(lib, "xcb_connect")?;
                let has_error: unsafe extern "C" fn(*mut c_void) -> c_int = sym(lib, "xcb_connection_has_error")?;
                let disconnect: unsafe extern "C" fn(*mut c_void) = sym(lib, "xcb_disconnect")?;
                let get_setup: unsafe extern "C" fn(*mut c_void) -> *const c_void = sym(lib, "xcb_get_setup")?;
                let roots: unsafe extern "C" fn(*const c_void) -> XcbScreenIterator = sym(lib, "xcb_setup_roots_iterator")?;
                let intern: unsafe extern "C" fn(*mut c_void, u8, u16, *const c_char) -> u32 = sym(lib, "xcb_intern_atom")?;
                let intern_reply: unsafe extern "C" fn(*mut c_void, u32, *mut *mut c_void) -> *mut c_void = sym(lib, "xcb_intern_atom_reply")?;
                let get_property = sym(lib, "xcb_get_property")?;
                let get_property_reply = sym(lib, "xcb_get_property_reply")?;
                let value = sym(lib, "xcb_get_property_value")?;
                let value_length = sym(lib, "xcb_get_property_value_length")?;
                let mut screen = 0;
                let conn = connect(std::ptr::null(), &mut screen);
                if conn.is_null() || has_error(conn) != 0 {
                    if !conn.is_null() {
                        disconnect(conn);
                    }
                    return None;
                }
                // The first screen's root: the game's (a Wine process uses the default display).
                let _ = screen;
                let it = roots(get_setup(conn));
                let root = if it.data.is_null() { 0 } else { *it.data };
                let atom = |name: &CStr| {
                    let cookie = intern(conn, 1, name.to_bytes().len() as u16, name.as_ptr());
                    let reply = intern_reply(conn, cookie, std::ptr::null_mut());
                    if reply.is_null() {
                        return 0;
                    }
                    // `xcb_intern_atom_reply_t`: response type, pad, sequence, length, then the atom.
                    let atom = *reply.cast::<u8>().add(8).cast::<u32>();
                    libc::free(reply);
                    atom
                };
                let (active, pid, client_list) = (atom(c"_NET_ACTIVE_WINDOW"), atom(c"_NET_WM_PID"), atom(c"_NET_CLIENT_LIST"));
                if root == 0 || active == 0 || pid == 0 {
                    disconnect(conn);
                    return None;
                }
                Some(Focus { lib, conn, root, active, pid, client_list, has_error, get_property, get_property_reply, value, value_length, disconnect })
            })()
        };
        if focus.is_none() {
            // SAFETY: the handle from the dlopen above, nothing of it in use.
            unsafe { libc::dlclose(lib) };
        }
        focus
    }

    /// One 32-bit value of `property` on `window`, `None` without it (or on any error).
    fn property(&self, window: u32, property: u32, kind: u32) -> Option<u32> {
        // SAFETY: a live connection; the reply is freed once read, and its value is read only when
        // it holds at least one 32-bit item.
        unsafe {
            let cookie = (self.get_property)(self.conn, 0, window, property, kind, 0, 1);
            let reply = (self.get_property_reply)(self.conn, cookie, std::ptr::null_mut());
            if reply.is_null() {
                return None;
            }
            let value = ((self.value_length)(reply) >= 1).then(|| *(self.value)(reply).cast::<u32>());
            libc::free(reply);
            value
        }
    }

    /// Whether the connection to the X server broke (it is then opened again, [`Poller::pressed`]).
    fn broken(&self) -> bool {
        // SAFETY: a connection of this object's, open until drop.
        unsafe { (self.has_error)(self.conn) != 0 }
    }

    /// Every 32-bit value of `property` on `window` (a list such as `_NET_CLIENT_LIST`), `None` without
    /// it (or on any error).
    fn property_list(&self, window: u32, property: u32, kind: u32) -> Option<Vec<u32>> {
        // SAFETY: as `property`; the reply's value holds `value_length` 32-bit items.
        unsafe {
            let cookie = (self.get_property)(self.conn, 0, window, property, kind, 0, 4096);
            let reply = (self.get_property_reply)(self.conn, cookie, std::ptr::null_mut());
            if reply.is_null() {
                return None;
            }
            let n = usize::try_from((self.value_length)(reply)).unwrap_or(0);
            let list = std::slice::from_raw_parts((self.value)(reply).cast::<u32>(), n).to_vec();
            libc::free(reply);
            Some(list)
        }
    }

    /// Whether the active window is this process's ([`focus_of`]): `None` when that cannot be told.
    fn game_focused(&self) -> Option<bool> {
        if self.broken() {
            return None;
        }
        let ours = std::process::id();
        let window = self.property(self.root, self.active, XCB_ATOM_WINDOW);
        let active_pid = window.filter(|&w| w != 0).and_then(|w| self.property(w, self.pid, XCB_ATOM_CARDINAL));
        // Whether the game owns any X window at all (a game on winewayland has none).
        let owns_x_window = self
            .property_list(self.root, self.client_list, XCB_ATOM_WINDOW)
            .map(|clients| clients.iter().any(|&w| self.property(w, self.pid, XCB_ATOM_CARDINAL) == Some(ours)));
        // Inside gamescope the nested server holds only the game: no active X window says nothing.
        if window == Some(0) && std::env::var_os("GAMESCOPE_WAYLAND_DISPLAY").is_some() {
            return None;
        }
        focus_of(window, active_pid, owns_x_window, ours)
    }
}

impl Drop for Focus {
    fn drop(&mut self) {
        // SAFETY: the connection is closed once, before its library is unloaded.
        unsafe {
            (self.disconnect)(self.conn);
            libc::dlclose(self.lib);
        }
    }
}

/// Whether the game (process `ours`) has the focus, from the active window (`None`: the server does
/// not say; `Some(0)`: no X window is active), that window's `_NET_WM_PID`, and whether any window of
/// `_NET_CLIENT_LIST` is the game's (`None`: no list). `Some(false)` only when another client is
/// positively known to have it: the active window is another process's, or none is active, while the
/// game has an X window of its own. A game with no X window (winewayland) cannot be told about:
/// `None`, and the key acts.
fn focus_of(window: Option<u32>, active_pid: Option<u32>, owns_x_window: Option<bool>, ours: u32) -> Option<bool> {
    if active_pid == Some(ours) {
        return Some(true);
    }
    // Another client is known to have the focus only against an X window of the game's own.
    if owns_x_window != Some(true) {
        return None;
    }
    match (window?, active_pid) {
        // No X window active while the game has one: something else (a Wayland window) has it.
        (0, _) => Some(false),
        (_, Some(_)) => Some(false),
        (_, None) => None,
    }
}

/// What a press of the toggle key does, given whether the game has the focus (`None`: unknown) and
/// whether the user asked for a global key (`NEURAL_FORGE_HOTKEY_GLOBAL=1`). Only a press known to
/// go to another client is ignored: where the focus cannot be told (gamescope's nested Xwayland, which
/// holds only the game; a bare X server without `_NET_ACTIVE_WINDOW`; no X server) the key acts, as it
/// always did.
fn press_acts(focused: Option<bool>, global: bool) -> bool {
    focused.is_none_or(|focused| focused || global)
}

/// How long after the X server could not be reached (or its connection broke) the focus check tries
/// to connect again.
const FOCUS_RETRY: Duration = Duration::from_secs(5);

// ---- the poller -----------------------------------------------------------------------------

enum Backend {
    Unopened,
    Evdev(Evdev),
    X11(X11),
    None,
}

pub struct Poller {
    backend: Backend,
    /// Presses of `pending_key` read but not yet answered (a poll is throttled, a press never lost).
    pending: u32,
    pending_key: u32,
    last_poll: Option<Instant>,
    /// The focus check, opened with the first press (`None` inside: there is no X server to ask), and
    /// when it was last opened (a broken or missing one is opened again after [`FOCUS_RETRY`]).
    focus: Option<Option<Focus>>,
    focus_opened: Option<Instant>,
}

// The backends hold raw descriptors/pointers used only from the layer's serialized present hook.
unsafe impl Send for Poller {}

impl Default for Poller {
    fn default() -> Self {
        Self { backend: Backend::Unopened, pending: 0, pending_key: 0, last_poll: None, focus: None, focus_opened: None }
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
            if key != self.pending_key {
                self.pending = 0;
                self.pending_key = key;
            }
            self.pending += match &mut self.backend {
                Backend::Evdev(e) => e.poll(key),
                Backend::X11(x) => x.poll(key),
                Backend::Unopened | Backend::None => 0,
            };
        }
        if self.pending == 0 || key != self.pending_key {
            return false;
        }
        self.pending -= 1;
        // Both backends see every press on the machine: the key acts only while the game has the
        // focus, or anywhere when the user asked for that.
        static GLOBAL: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| neural_forge_protocol::env::flag("NEURAL_FORGE_HOTKEY_GLOBAL"));
        let now = Instant::now();
        // A connection that broke, or an X server that could not be reached, is tried again (not on
        // every press: at most every FOCUS_RETRY).
        let stale = self.focus.as_ref().is_some_and(|f| f.as_ref().is_none_or(Focus::broken));
        if stale && self.focus_opened.is_none_or(|t| now.duration_since(t) >= FOCUS_RETRY) {
            self.focus = None;
        }
        let opened = &mut self.focus_opened;
        let focus = self.focus.get_or_insert_with(|| {
            *opened = Some(now);
            Focus::open()
        });
        let focused = focus.as_ref().and_then(Focus::game_focused);
        let acts = press_acts(focused, *GLOBAL);
        if focused.is_none() {
            static SAID: AtomicBool = AtomicBool::new(false);
            if !SAID.swap(true, Ordering::Relaxed) {
                crate::log!(
                    "[hotkey] the toggle key was pressed where the game's focus cannot be told (gamescope, no X server, or no \
                     _NET_ACTIVE_WINDOW): it acts"
                );
            }
        }
        acts
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
        assert_eq!(EVIOCGVERSION, 0x8004_4501);
        assert_eq!(eviocgbit(1, 96) as u64, 0x8060_4521);
    }

    #[test]
    fn a_press_seen_by_one_drain_is_delivered_once() {
        let mut p = Poller { backend: Backend::None, ..Poller::default() };
        p.pending = 1;
        p.pending_key = 87;
        p.last_poll = Some(Instant::now());
        // No X server to ask about the focus here: the press is consumed, and acts (unknown focus).
        p.focus = Some(None);
        assert!(!p.pressed(88), "another key's press is never one of these");
        assert_eq!(p.pending, 1);
        assert_eq!(p.pressed(87), press_acts(None, neural_forge_protocol::env::flag("NEURAL_FORGE_HOTKEY_GLOBAL")));
        assert_eq!(p.pending, 0, "delivered once");
        assert!(!p.pressed(87));
    }

    /// Serializes the tests that use the process-wide X connection table and handler chain.
    static X11_STATE: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// An I/O error on one of the layer's own X connections marks that one dead and never reaches
    /// the game's handler; one on any other connection goes to the handler that was installed before.
    #[test]
    fn io_errors_on_other_connections_go_to_the_previous_handler() {
        let _state = X11_STATE.lock().unwrap_or_else(|e| e.into_inner());
        static SEEN: AtomicUsize = AtomicUsize::new(0);
        unsafe extern "C" fn previous(display: *mut c_void) -> c_int {
            SEEN.store(display as usize, Ordering::SeqCst);
            7
        }
        let (first, second) = (0x1000 as *mut c_void, 0x1800 as *mut c_void);
        let a = register_display(first).expect("a free entry");
        let b = register_display(second).expect("a second device's poller has its own entry");
        PREVIOUS_IO_ERROR.store(previous as *const () as usize, Ordering::SeqCst);
        assert_eq!(unsafe { io_error(0x2000 as *mut c_void) }, 7);
        assert_eq!(SEEN.load(Ordering::SeqCst), 0x2000);
        assert!(!X11_DEAD[a].load(Ordering::SeqCst) && !X11_DEAD[b].load(Ordering::SeqCst));
        assert_eq!(unsafe { io_error(first) }, 0);
        assert!(X11_DEAD[a].load(Ordering::SeqCst), "our connection is marked dead");
        assert!(!X11_DEAD[b].load(Ordering::SeqCst), "the other one of ours lives on");
        assert_eq!(unsafe { io_error(second) }, 0, "the second connection is ours too");
        assert_eq!(SEEN.load(Ordering::SeqCst), 0x2000, "the game's handler never sees either");
        unsafe { io_error_exit(second, b as *mut c_void) };
        assert!(X11_DEAD[b].load(Ordering::SeqCst), "the exit handler marks its own entry");
        release_display(a);
        release_display(b);
        assert_eq!(unsafe { io_error(first) }, 7, "a released connection is no longer the layer's");
        PREVIOUS_IO_ERROR.store(0, Ordering::SeqCst);
    }

    /// Our own handler is never kept as the one before it: a second `XSetIOErrorHandler` returns it,
    /// and chaining to it made an error on any other connection recurse until the stack overflowed.
    #[test]
    fn our_own_handler_is_never_chained_to_itself() {
        let _state = X11_STATE.lock().unwrap_or_else(|e| e.into_inner());
        unsafe extern "C" fn wine(_display: *mut c_void) -> c_int {
            3
        }
        chain_previous(Some(wine));
        chain_previous(Some(io_error));
        assert_eq!(PREVIOUS_IO_ERROR.load(Ordering::SeqCst), wine as *const () as usize, "the handler from before ours is kept");
        assert_eq!(unsafe { io_error(0x2000 as *mut c_void) }, 3, "another connection's error reaches it, once");
        chain_previous(None);
        assert_eq!(unsafe { io_error(0x2000 as *mut c_void) }, 0, "with none before ours the error simply returns");
        PREVIOUS_IO_ERROR.store(0, Ordering::SeqCst);
    }

    /// All entries taken: a further connection is refused, not mistaken for another's.
    #[test]
    fn the_connection_table_refuses_a_connection_past_its_size() {
        let _state = X11_STATE.lock().unwrap_or_else(|e| e.into_inner());
        let taken: Vec<usize> = (1..=X11_SLOTS).map(|i| register_display((0x4000 + i * 0x100) as *mut c_void).expect("free")).collect();
        assert_eq!(register_display(0x9000 as *mut c_void), None);
        for slot in taken {
            release_display(slot);
        }
        let again = register_display(0x9000 as *mut c_void).expect("released entries are reused");
        release_display(again);
    }

    /// A press acts with the game focused; never with another window focused unless global was asked
    /// for; and where focus cannot be told, always.
    #[test]
    fn a_press_acts_only_with_the_game_focused_unless_global() {
        assert!(press_acts(Some(true), false));
        assert!(!press_acts(Some(false), false));
        assert!(press_acts(Some(false), true));
        assert!(press_acts(None, false), "unknown focus (gamescope's nested server holds only the game): acts");
    }

    /// The focus is "another client's" only when that is positively known: the game owns an X window
    /// and another process's window (or no X window) is active.
    #[test]
    fn the_focus_is_anothers_only_when_the_game_has_an_x_window() {
        let (ours, other) = (100, 200);
        // winewayland: the game has no X window, another X window is active: unknown, the key acts.
        assert_eq!(focus_of(Some(7), Some(other), Some(false), ours), None);
        assert!(press_acts(focus_of(Some(7), Some(other), Some(false), ours), false));
        assert_eq!(focus_of(Some(0), None, Some(false), ours), None, "nothing active, no X window of the game's");
        assert_eq!(focus_of(Some(7), Some(other), None, ours), None, "no client list: cannot tell");
        // The game has an X window and another app is active: ignored.
        assert_eq!(focus_of(Some(7), Some(other), Some(true), ours), Some(false));
        assert!(!press_acts(focus_of(Some(7), Some(other), Some(true), ours), false));
        assert_eq!(focus_of(Some(0), None, Some(true), ours), Some(false), "a Wayland window is active");
        // The game's window is active: acts.
        assert_eq!(focus_of(Some(9), Some(ours), Some(true), ours), Some(true));
        assert!(press_acts(focus_of(Some(9), Some(ours), Some(true), ours), false));
        assert_eq!(focus_of(Some(8), None, Some(true), ours), None, "an active window without a PID");
        assert_eq!(focus_of(None, None, Some(true), ours), None, "no _NET_ACTIVE_WINDOW");
        assert!(press_acts(None, true));
    }
}
