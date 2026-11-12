//! The model on a second GPU (docs/NATIVE_BACKEND.md, "The model on a second GPU"): the after-the-upscaler path's
//! network frames run on a device of another GPU (or, to test the path, a second device of the game's own GPU),
//! while the game's GPU only captures the finished frame and composes the answer.
//!
//! The frame already crosses system memory on that path (the proxy and answer regions of the shared memory), so
//! the server ([`super::native_post::PostServer`]) runs unchanged on the other device: it imports the regions as
//! host memory there (`VK_EXT_external_memory_host`) where the device allows, and stages them otherwise. The
//! device is made through the loader's `VK_LOADER_LAYER_CREATE_DEVICE_CALLBACK` (LoaderLayerInterface.md, "Layers
//! that create devices"), so the layers below this one see it as a device of their own and this layer's hooks
//! never do.
//!
//! `NEURAL_FORGE_OFFLOAD_GPU` names the device ([`Choice`]). The device must have everything the network needs
//! (`neural_forge_native::device_supported`); anything else, a failed setup, the network not running there
//! ([`ServerRule`]), frames too slow for the game ([`LateRule`]), or a frame that did not finish (once it has), sends the
//! network back to the game's GPU for the rest of the session ([`given_up`], `super::native_post`), said once in the log. Only the finished-frame placement can move: the holds before the upscaler and after DLSS
//! run inside DLSS's own command buffer on the game's GPU.
//!
//! `NEURAL_FORGE_OFFLOAD_STANDIN=1` (a test switch, never for a game session) runs a copy in place of the network
//! on the other device, so that two real, different GPUs (one that cannot run the network) exercise the copies,
//! the waits and the fall-back; every answer is then the frame itself.

use std::ffi::{c_char, c_void, CStr};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use ash::vk;

use super::native::Setup;

/// The frames the present judges the other GPU over ([`LateRule`]), and how many of them may come late.
pub(crate) const LATE_WINDOW: usize = 120;
pub(crate) const LATE_ALLOWED: usize = 30;
/// A frame is late when its answer takes longer than this many present intervals from its request (the game's own time
/// between presents; the present waits for the rest) and longer than [`LATE_FLOOR`], or does not come within the
/// present's budget. The network costs the game's GPU its own time too when it runs there, which the present does not
/// see: a second GPU somewhat slower than the game's frame (40 ms against a 120 fps game) still beats running the
/// network after the game's frame on its GPU, so a frame is judged late only well past both.
pub(crate) const LATE_INTERVALS: u32 = 3;
pub(crate) const LATE_FLOOR: Duration = Duration::from_millis(100);
/// This many sent frames in a row with no answer within the present's budget fall back at once (each held the game up
/// to that budget).
pub(crate) const UNANSWERED_IN_A_ROW: u32 = 5;
/// A single frame of the other GPU slower than this falls back at once (vkDaedalus's rule for its second GPU: a
/// second), except the first frames after a build ([`WARM_FRAMES`]: the driver compiles and the first copies map).
pub(crate) const HUNG: Duration = Duration::from_secs(1);
pub(crate) const WARM_FRAMES: u32 = 10;
/// The network not running on the other GPU for this long while frames come (it could not be built, the VRAM guard
/// paused it) falls back: the frames meanwhile go out without the model.
pub(crate) const NOT_RUNNING: Duration = Duration::from_secs(30);

/// Which device runs the network (`NEURAL_FORGE_OFFLOAD_GPU`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Choice {
    /// The game's own device (unset, empty or `off`).
    Off,
    /// A second device of the game's own GPU, through the whole offload path (`same`): a test of the path; it
    /// saves nothing.
    Same,
    /// The device at this index of the instance's list (as the log lists them at start).
    Index(u32),
    /// The device with this PCI vendor and device ID (hexadecimal, `10de:2704`), and where two cards share them, at
    /// this PCI address (`10de:2704@0000:01:00.0`; the domain may be left out).
    Pci { vendor: u32, device: u32, bus: Option<PciAddress> },
}

/// A PCI address (`VK_EXT_pci_bus_info`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PciAddress {
    pub domain: u32,
    pub bus: u32,
    pub device: u32,
    pub function: u32,
}

impl std::fmt::Display for PciAddress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:04x}:{:02x}:{:02x}.{:x}", self.domain, self.bus, self.device, self.function)
    }
}

impl PciAddress {
    /// `0000:01:00.0` or `01:00.0` (the domain 0): hexadecimal, at most 4 digits for the domain, 2 for the bus and
    /// the device (at most 0x1f) and 1 for the function (at most 7).
    fn parse(s: &str) -> Option<Self> {
        let (rest, function) = s.rsplit_once('.')?;
        let parts: Vec<&str> = rest.split(':').collect();
        let (domain, bus, device) = match parts.as_slice() {
            [d, b, v] => (Some(*d), *b, *v),
            [b, v] => (None, *b, *v),
            _ => return None,
        };
        let domain = match domain {
            Some(d) => hex(d, 4)?,
            None => 0,
        };
        let (bus, device, function) = (hex(bus, 2)?, hex(device, 2)?, hex(function, 1)?);
        (device <= 0x1f && function <= 7).then_some(Self { domain, bus, device, function })
    }
}

/// A hexadecimal number of 1 to `digits` digits, nothing else (no sign, no `0x`).
fn hex(t: &str, digits: usize) -> Option<u32> {
    (!t.is_empty() && t.len() <= digits && t.bytes().all(|b| b.is_ascii_hexdigit())).then(|| u32::from_str_radix(t, 16).ok()).flatten()
}

impl Choice {
    /// The variable's value: `off` (or nothing), `same`, an index (decimal digits), or `vendor:device[@pci-address]`
    /// (each ID 4 hexadecimal digits).
    pub(crate) fn parse(value: &str) -> Result<Self, String> {
        let v = value.trim().to_ascii_lowercase();
        if v.is_empty() || v == "off" {
            return Ok(Self::Off);
        }
        if v == "same" {
            return Ok(Self::Same);
        }
        if v.bytes().all(|b| b.is_ascii_digit()) {
            return v.parse::<u32>().map(Self::Index).map_err(|_| format!("{value:?}: the index is too large"));
        }
        let (ids, bus) = match v.split_once('@') {
            Some((ids, bus)) => (ids, Some(PciAddress::parse(bus).ok_or_else(|| format!("{bus:?} is no PCI address (0000:01:00.0)"))?)),
            None => (v.as_str(), None),
        };
        let (vendor, device) = ids.split_once(':').ok_or_else(|| format!("{value:?} is not off, same, an index or vendor:device[@pci-address]"))?;
        let id = |t: &str| hex(t, 4).ok_or_else(|| format!("{t:?} is not a PCI ID (4 hexadecimal digits, as 10de)"));
        Ok(Self::Pci { vendor: id(vendor)?, device: id(device)?, bus })
    }

    /// `NEURAL_FORGE_OFFLOAD_GPU`, read once; a value that does not parse is logged and means [`Self::Off`].
    pub(crate) fn current() -> Self {
        static CHOICE: std::sync::LazyLock<Choice> = std::sync::LazyLock::new(|| {
            let Some(value) = neural_forge_protocol::env::var("NEURAL_FORGE_OFFLOAD_GPU") else { return Choice::Off };
            Choice::parse(&value).unwrap_or_else(|why| {
                crate::log!("[offload] NEURAL_FORGE_OFFLOAD_GPU: {why}; the model stays on the game's GPU");
                Choice::Off
            })
        });
        CHOICE.clone()
    }
}

/// `NEURAL_FORGE_OFFLOAD_STANDIN=1`: a copy in place of the network on the other device (a test of the path).
pub(crate) fn standin() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| neural_forge_protocol::env::flag("NEURAL_FORGE_OFFLOAD_STANDIN"));
    *ON
}

/// What the offload is doing, for the Status tab and `doctor` (`shm`'s layer status) and the log.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum State {
    /// Not asked for (or not at the finished-frame placement).
    Off,
    /// The device is being made and the network loaded (the GPU's name).
    Starting(String),
    /// The network runs on the other GPU (its name).
    Active(String),
    /// A frame on the other GPU (its name) did not finish: the server waits for it, the frames go out without the model.
    Stalled(String),
    /// Back on the game's GPU for the session, and why.
    FellBack(String),
}

static STATE: Mutex<State> = Mutex::new(State::Off);

/// Notes what the offload is doing ([`State`]).
pub(crate) fn note_state(state: State) {
    let mut now = STATE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if let (State::FellBack(_), State::Starting(_) | State::Active(_) | State::Stalled(_)) = (&*now, &state) {
        // A fall-back is for the session.
        return;
    }
    *now = state;
}

/// The offload was given up for the session: every device of the process runs the network on its own GPU from then on.
pub(crate) fn given_up() -> bool {
    matches!(state(), State::FellBack(_))
}

/// What the offload is doing.
pub(crate) fn state() -> State {
    STATE.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone()
}

/// The Status tab's line for it, or `None` while it is off.
pub(crate) fn status_line() -> Option<String> {
    match state() {
        State::Off => None,
        State::Starting(gpu) => Some(format!("second GPU: starting on {gpu}")),
        State::Active(gpu) => Some(format!("second GPU: the model runs on {gpu}, each frame shown one frame later")),
        State::Stalled(gpu) => Some(format!("second GPU: stalled: a frame on {gpu} did not finish; waiting for it, the frames go out without the model")),
        State::FellBack(why) => Some(format!("second GPU: given up ({why}); the model runs on the game's GPU")),
    }
}

/// Whether frames are offloaded now: the server on the other device is serving. The present delays every frame by
/// one while it is (`capture`): the game's GPU renders the next frame while the other GPU runs this one's network.
static ACTIVE: AtomicBool = AtomicBool::new(false);

pub(crate) fn active() -> bool {
    ACTIVE.load(Ordering::Acquire)
}

pub(crate) fn set_active(on: bool) {
    if on {
        // A server starting afresh is judged afresh.
        *LATE.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = None;
        take_give_up();
    }
    ACTIVE.store(on, Ordering::Release);
}

/// One physical device of the instance, as the choice matches it.
#[derive(Clone, Debug)]
pub(crate) struct Candidate {
    pub index: u32,
    pub vendor: u32,
    pub device: u32,
    pub pci: Option<PciAddress>,
    pub game: bool,
}

/// Which of `candidates` the choice names (an index into the slice), or why none.
pub(crate) fn pick(choice: &Choice, candidates: &[Candidate]) -> Result<usize, String> {
    let found = match choice {
        Choice::Off => return Err("no offload asked for".into()),
        Choice::Same => candidates.iter().position(|c| c.game),
        Choice::Index(i) => candidates.iter().position(|c| c.index == *i),
        Choice::Pci { vendor, device, bus } => {
            let matching: Vec<usize> = (0..candidates.len())
                .filter(|&k| candidates[k].vendor == *vendor && candidates[k].device == *device && bus.is_none_or(|b| candidates[k].pci == Some(b)))
                .collect();
            if matching.len() > 1 {
                // Two of the same card: the one the game does not run on, else name the address.
                let others: Vec<usize> = matching.iter().copied().filter(|&k| !candidates[k].game).collect();
                if others.len() == 1 {
                    return Ok(others[0]);
                }
                return Err(format!("{} devices are {vendor:04x}:{device:04x}: add the PCI address (@0000:01:00.0)", matching.len()));
            }
            matching.first().copied()
        }
    };
    let k = found.ok_or_else(|| format!("no device matches {choice:?}"))?;
    if candidates[k].game && *choice != Choice::Same {
        return Err("that is the game's own GPU (`same` tests the path on it)".into());
    }
    Ok(k)
}

type PfnLayerCreateDevice = unsafe extern "system" fn(
    vk::Instance,
    vk::PhysicalDevice,
    *const vk::DeviceCreateInfo,
    *const vk::AllocationCallbacks,
    *mut vk::Device,
    vk::PFN_vkGetInstanceProcAddr,
    *mut Option<vk::PFN_vkGetDeviceProcAddr>,
) -> vk::Result;
type PfnLayerDestroyDevice = unsafe extern "system" fn(vk::Device, *const vk::AllocationCallbacks, vk::PFN_vkDestroyDevice);

/// The loader's `VkLayerInstanceCreateInfo` with `function == VK_LOADER_LAYER_CREATE_DEVICE_CALLBACK` (`vk_layer.h`).
#[repr(C)]
struct LoaderCreateDeviceCallback {
    s_type: vk::StructureType,
    p_next: *const c_void,
    function: i32,
    create: Option<PfnLayerCreateDevice>,
    destroy: Option<PfnLayerDestroyDevice>,
}

const LOADER_INSTANCE_CREATE_INFO: vk::StructureType = vk::StructureType::from_raw(47);
const LAYER_CREATE_DEVICE_CALLBACK: i32 = 2;

#[derive(Clone, Copy)]
struct Callbacks {
    create: PfnLayerCreateDevice,
    destroy: PfnLayerDestroyDevice,
}

/// Each instance's device callbacks, from its creation.
static CALLBACKS: Mutex<Vec<(vk::Instance, Callbacks)>> = Mutex::new(Vec::new());

/// Notes the loader's device callbacks from `info`'s chain (the instance's creation) for `instance`.
///
/// # Safety
/// `info`'s chain is valid.
pub(crate) unsafe fn note_instance(instance: vk::Instance, info: &vk::InstanceCreateInfo) {
    let mut next = info.p_next.cast::<LoaderCreateDeviceCallback>();
    let mut found = None;
    while !next.is_null() {
        // SAFETY: every node of a valid chain starts with sType and pNext; the rest is read on the loader's node only.
        let node = unsafe { &*next };
        if node.s_type == LOADER_INSTANCE_CREATE_INFO && node.function == LAYER_CREATE_DEVICE_CALLBACK {
            if let (Some(create), Some(destroy)) = (node.create, node.destroy) {
                found = Some(Callbacks { create, destroy });
            }
            break;
        }
        next = node.p_next.cast();
    }
    let mut all = CALLBACKS.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    all.retain(|(i, _)| *i != instance);
    if let Some(c) = found {
        all.push((instance, c));
    }
}

fn callbacks(instance: vk::Instance) -> Option<Callbacks> {
    CALLBACKS.lock().unwrap_or_else(std::sync::PoisonError::into_inner).iter().find(|(i, _)| *i == instance).map(|(_, c)| *c)
}

thread_local! {
    /// Set while this thread makes the offload device: should the loader reach this layer's own `vkCreateDevice`
    /// anyway, it passes the request on untouched.
    static MAKING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether this thread is making the offload device ([`MAKING`]).
pub(crate) fn making() -> bool {
    MAKING.with(std::cell::Cell::get)
}

/// This layer's own `vkGetInstanceProcAddr`, as the loader found it (its manifest names no other): the loader
/// builds the new device's chain from the layer below the one this names.
fn own_gipa() -> Option<vk::PFN_vkGetInstanceProcAddr> {
    let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
    // SAFETY: an address inside this library; `info` is written by the call.
    if unsafe { libc::dladdr(own_gipa as *const c_void, &mut info) } == 0 || info.dli_fname.is_null() {
        return None;
    }
    // SAFETY: the path dladdr gave; RTLD_NOLOAD only finds this library, already loaded.
    let handle = unsafe { libc::dlopen(info.dli_fname, libc::RTLD_NOW | libc::RTLD_NOLOAD) };
    if handle.is_null() {
        return None;
    }
    // SAFETY: a symbol lookup in the handle just opened; closed after (the library stays loaded).
    let symbol = unsafe { libc::dlsym(handle, c"vkGetInstanceProcAddr".as_ptr()) };
    unsafe { libc::dlclose(handle) };
    // SAFETY: the layer's exported vkGetInstanceProcAddr has this signature.
    (!symbol.is_null()).then(|| unsafe { std::mem::transmute::<*mut c_void, vk::PFN_vkGetInstanceProcAddr>(symbol) })
}

/// What the log says about a physical device.
struct Described {
    name: String,
    vendor: u32,
    device: u32,
    pci: Option<PciAddress>,
    driver: String,
}

/// `core2`: the instance may call `vkGetPhysicalDeviceProperties2` (Vulkan 1.1 or later); without it only the plain
/// properties are read (no PCI address, the driver as its version number).
///
/// # Safety
/// `physical` belongs to `instance`.
unsafe fn describe(instance: &ash::Instance, physical: vk::PhysicalDevice, core2: bool) -> Described {
    // SAFETY: as the contract.
    let props = unsafe { instance.get_physical_device_properties(physical) };
    let text = |chars: &[c_char]| unsafe { CStr::from_ptr(chars.as_ptr()) }.to_string_lossy().into_owned();
    if !core2 {
        return Described {
            name: text(&props.device_name),
            vendor: props.vendor_id,
            device: props.device_id,
            pci: None,
            driver: format!("driver version {:#x}", props.driver_version),
        };
    }
    // SAFETY: as the contract.
    let has_pci = unsafe { instance.enumerate_device_extension_properties(physical) }
        .unwrap_or_default()
        .iter()
        .any(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) } == c"VK_EXT_pci_bus_info");
    let mut pci = vk::PhysicalDevicePCIBusInfoPropertiesEXT::default();
    let mut driver = vk::PhysicalDeviceDriverProperties::default();
    // The driver's name and version: core in Vulkan 1.2 devices.
    let has_driver = props.api_version >= vk::API_VERSION_1_2;
    {
        let mut props2 = vk::PhysicalDeviceProperties2::default();
        if has_driver {
            props2 = props2.push_next(&mut driver);
        }
        if has_pci {
            props2 = props2.push_next(&mut pci);
        }
        // SAFETY: Vulkan 1.1 core (`core2`); the chain holds only what the device offers, and outlives the call.
        unsafe { instance.get_physical_device_properties2(physical, &mut props2) };
    }
    Described {
        name: text(&props.device_name),
        vendor: props.vendor_id,
        device: props.device_id,
        pci: has_pci.then_some(PciAddress { domain: pci.pci_domain, bus: pci.pci_bus, device: pci.pci_device, function: pci.pci_function }),
        driver: if has_driver {
            format!("{} {}", text(&driver.driver_name), text(&driver.driver_info)).trim().to_string()
        } else {
            format!("driver version {:#x}", props.driver_version)
        },
    }
}

impl Described {
    fn line(&self) -> String {
        format!(
            "{} [{:04x}:{:04x}{}], {}",
            self.name,
            self.vendor,
            self.device,
            self.pci.map_or(String::new(), |p| format!("@{p}")),
            if self.driver.is_empty() { "driver unknown".to_string() } else { self.driver.clone() }
        )
    }
}

/// The device the network runs on, made for it.
pub(crate) struct Offload {
    pub device: ash::Device,
    pub physical: vk::PhysicalDevice,
    pub setup: Setup,
    /// The shared-memory regions are imported as host memory on it (else staged).
    pub import: bool,
    /// The stand-in copy runs in place of the network ([`standin`]).
    pub standin: bool,
    /// For the log: the device's name, IDs, PCI address and driver.
    pub name: String,
    /// For the log when the server starts on it: which GPU, the game's, and how the frame crosses.
    pub plan: String,
    destroy: PfnLayerDestroyDevice,
    _registration: crate::loader_data::Registration,
}

/// The queue family the network's two queues come from: compute without graphics with two queues, else any compute
/// family with two (the stand-in needs one).
fn family(families: &[vk::QueueFamilyProperties], wanted: u32) -> Option<u32> {
    let fits = |f: &vk::QueueFamilyProperties| f.queue_flags.contains(vk::QueueFlags::COMPUTE) && f.queue_count >= wanted;
    let pick =
        |no_graphics: bool| (0..families.len()).find(|&i| fits(&families[i]) && (!no_graphics || !families[i].queue_flags.contains(vk::QueueFlags::GRAPHICS)));
    pick(true).or_else(|| pick(false)).map(|i| i as u32)
}

impl Offload {
    /// Makes the device `choice` names on the game's instance (`game`: the setup of the game's device being made from
    /// `game_info`). Called inside the game's `vkCreateDevice` ([`make_for`]): the loader's device callback needs the
    /// lock its trampoline holds there.
    ///
    /// # Safety
    /// `instance` is the game's (the next layer's dispatch), live for the call; `game_info` is the game's request, its
    /// chain the loader's.
    unsafe fn create(instance: &ash::Instance, game: Setup, game_info: &vk::DeviceCreateInfo, choice: &Choice) -> Result<Self, String> {
        let callbacks = callbacks(game.instance).ok_or("the loader gave this instance no device callback (a Vulkan loader older than 1.2.x)")?;
        let layer_gipa = own_gipa().ok_or("the layer could not find its own vkGetInstanceProcAddr")?;
        // SAFETY: the game's live instance.
        let physicals = unsafe { instance.enumerate_physical_devices() }.map_err(|e| format!("listing the GPUs failed ({e:?})"))?;
        // SAFETY: physical devices of this 1.3 instance (the game's device runs the network).
        let described: Vec<Described> = physicals.iter().map(|&p| unsafe { describe(instance, p, true) }).collect();
        let candidates: Vec<Candidate> = described
            .iter()
            .enumerate()
            .map(|(i, d)| Candidate { index: i as u32, vendor: d.vendor, device: d.device, pci: d.pci, game: physicals[i] == game.physical })
            .collect();
        let k = pick(choice, &candidates)?;
        let physical = physicals[k];
        let same = physical == game.physical;
        let standin = standin();
        // SAFETY: a physical device of this 1.3 instance (the game's device runs the network, so it is 1.3).
        let supported = unsafe { neural_forge_native::device_supported(game.gipa, game.instance, vk::API_VERSION_1_3, physical) };
        if let Err(why) = &supported {
            if !standin {
                return Err(format!("{} cannot run the network: {why}", described[k].name));
            }
            crate::log!("[offload] {} cannot run the network ({why}): the stand-in copy runs there instead (NEURAL_FORGE_OFFLOAD_STANDIN)", described[k].name);
        }
        let run_network = supported.is_ok() && !standin;
        // SAFETY: as above.
        let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
        let queues_wanted = if run_network { 2 } else { 1 };
        let family = family(&families, queues_wanted).ok_or_else(|| format!("{} has no compute family with {queues_wanted} queue(s)", described[k].name))?;
        let priorities = [1.0f32; 2];
        let queue = [vk::DeviceQueueCreateInfo::default().queue_family_index(family).queue_priorities(&priorities[..queues_wanted as usize])];
        // SAFETY: as above.
        let extensions = unsafe { instance.enumerate_device_extension_properties(physical) }.unwrap_or_default();
        let offers = |name: &CStr| extensions.iter().any(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) } == name);
        let host = offers(crate::EXTERNAL_MEMORY_HOST_EXTENSION);
        let names: Vec<*const c_char> = if host { vec![crate::EXTERNAL_MEMORY_HOST_EXTENSION.as_ptr()] } else { Vec::new() };
        let base = vk::DeviceCreateInfo::default().queue_create_infos(&queue).enabled_extension_names(&names);
        // The network's extensions and features on top (they outlive the call).
        let extension = if run_network {
            // SAFETY: the next layer's gipa for this 1.3 instance; `base` outlives the extension.
            Some(unsafe { neural_forge_native::device_extend(game.gipa, game.instance, vk::API_VERSION_1_3, physical, &base) }?)
        } else {
            None
        };
        let info = extension.as_ref().map_or(&base, |e| &e.info);
        let mut device = vk::Device::null();
        let mut next_gdpa: Option<vk::PFN_vkGetDeviceProcAddr> = None;
        MAKING.with(|m| m.set(true));
        // SAFETY: the loader's callback for this instance, with a valid request; this layer's gipa as the caller.
        let made = unsafe { (callbacks.create)(game.instance, physical, info, std::ptr::null(), &mut device, layer_gipa, &mut next_gdpa) };
        MAKING.with(|m| m.set(false));
        drop(extension);
        if made != vk::Result::SUCCESS {
            return Err(format!("making the device on {} failed ({made:?})", described[k].name));
        }
        let Some(gdpa) = next_gdpa else {
            // Destroyed again through the loader, with the vkDestroyDevice the next layer's instance gives for it.
            // SAFETY: just made, nothing of it used; the name is a valid C string.
            if let Some(destroy) = unsafe { instance.get_device_proc_addr(device, c"vkDestroyDevice".as_ptr()) } {
                // SAFETY: the loader's callback with the device's own vkDestroyDevice (the PFN's signature by its name).
                unsafe { (callbacks.destroy)(device, std::ptr::null(), std::mem::transmute::<unsafe extern "system" fn(), vk::PFN_vkDestroyDevice>(destroy)) };
            }
            return Err("the loader made the device without the next vkGetDeviceProcAddr".into());
        };
        // SAFETY: the next layer's vkGetDeviceProcAddr for the device just made.
        let ash_device =
            unsafe { ash::Device::load_with(|name| std::mem::transmute::<vk::PFN_vkVoidFunction, *const c_void>((gdpa)(device, name.as_ptr())), device) };
        // Objects the layer gets from below the loader for this device get the loader's dispatch pointer through the
        // game's device's callback (the loader's own, which finds any device it made).
        // SAFETY: the game's create info, its chain the loader's (its device-loader-data callback serves every device the
        // loader made).
        let registration = unsafe { crate::loader_data::register(device, game_info) };
        crate::note_secondary_device(device, true);
        let setup = Setup {
            gipa: game.gipa,
            instance: game.instance,
            physical,
            frame_family: family,
            family,
            index: 0,
            post_index: if run_network { 1 } else { 0 },
            properties2: game.properties2,
            flow: None,
            shared: None,
            bda_ext: false,
        };
        let name = described[k].line();
        crate::log!("[offload] a device made on GPU {k} ({name}) for the model on the finished frame: used once Placement is Finished frame");
        crate::logging::flush();
        let plan = format!(
            "the network runs on GPU {k}: {name}{}; the game's GPU: {}; queue family {family}; the frame crosses system memory: {} on the other GPU, host copies on the game's",
            if same { " (the game's own GPU, a second device: a test of the path, it saves nothing)" } else { "" },
            described.iter().zip(&physicals).find(|(_, &p)| p == game.physical).map_or("unknown".to_string(), |(d, _)| d.line()),
            if host { "imported host memory (VK_EXT_external_memory_host, no CPU copy)" } else { "staged copies (no VK_EXT_external_memory_host)" },
        );
        Ok(Self {
            device: ash_device,
            physical,
            setup,
            import: host,
            standin: !run_network,
            name,
            plan,
            destroy: callbacks.destroy,
            _registration: registration,
        })
    }

    /// Destroys the device (through the loader, as it was made).
    ///
    /// # Safety
    /// Nothing of the device's is pending or alive.
    pub(crate) unsafe fn destroy(self) {
        let destroy_device = self.device.fp_v1_0().destroy_device;
        // SAFETY: as the contract; the loader's callback with the device's own vkDestroyDevice.
        unsafe { (self.destroy)(self.device.handle(), std::ptr::null(), destroy_device) };
        crate::note_secondary_device(self.device.handle(), false);
    }
}

/// The GPUs of the instance, in the log once per process (where there is more than one): the indices and IDs
/// `NEURAL_FORGE_OFFLOAD_GPU` takes.
///
/// # Safety
/// `game` is a physical device of `instance`; `core2` only where the instance has Vulkan 1.1.
pub(crate) unsafe fn log_gpus(instance: &ash::Instance, game: vk::PhysicalDevice, core2: bool) {
    static SAID: AtomicBool = AtomicBool::new(false);
    if SAID.load(Ordering::Relaxed) {
        return;
    }
    // SAFETY: the game's live instance.
    let Ok(physicals) = (unsafe { instance.enumerate_physical_devices() }) else { return };
    // Where the variable is set, or there is more than one real GPU (a CPU renderer such as llvmpipe does not count).
    // SAFETY: plain property queries on the instance's own physical devices.
    let gpus = physicals.iter().filter(|&&p| unsafe { instance.get_physical_device_properties(p) }.device_type != vk::PhysicalDeviceType::CPU).count();
    if (gpus < 2 && Choice::current() == Choice::Off) || SAID.swap(true, Ordering::Relaxed) {
        return;
    }
    for (i, &p) in physicals.iter().enumerate() {
        // SAFETY: as the contract.
        let d = unsafe { describe(instance, p, core2) };
        crate::log!(
            "[offload] GPU {i}: {}{} (NEURAL_FORGE_OFFLOAD_GPU={i} or {:04x}:{:04x} runs the model there)",
            d.line(),
            if p == game { ", the game's" } else { "" },
            d.vendor,
            d.device
        );
    }
}

/// The offload devices made with the game's devices ([`make_for`]): per game device, the device (or why none), and
/// whether a frame on it may still run (it is then never destroyed).
struct Made {
    game: vk::Device,
    offload: Option<Result<Offload, String>>,
    busy: bool,
}

// SAFETY: the handles and function pointers inside are plain values; the device is used by one thread at a time (the
// server's, or the teardown's once the server was joined).
unsafe impl Send for Made {}

static MADE: Mutex<Vec<Made>> = Mutex::new(Vec::new());

/// Called inside the game's `vkCreateDevice` (the loader holds its lock there, which its device callbacks need: they
/// change the loader's device lists without taking it themselves), for a device the network runs on (`game`, made
/// from `info`): logs the instance's GPUs, and with `NEURAL_FORGE_OFFLOAD_GPU` set makes the offload device now. The
/// server takes it later ([`take`]).
///
/// # Safety
/// Called inside `vkCreateDevice` for `device`; `info` is its request (the loader's chain); `game` its setup.
pub(crate) unsafe fn make_for(instance: &ash::Instance, game: Setup, device: vk::Device, info: &vk::DeviceCreateInfo) {
    let choice = Choice::current();
    // Given up for the process: no more devices for it.
    if choice == Choice::Off || given_up() {
        return;
    }
    // SAFETY: as the contract.
    let made = unsafe { Offload::create(instance, game, info, &choice) };
    if let Err(why) = &made {
        crate::log!("[offload] the model stays on the game's GPU: {why}");
        crate::logging::flush();
    }
    let mut all = MADE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    all.retain(|m| m.game != device);
    all.push(Made { game: device, offload: Some(made), busy: false });
}

/// The offload device made with `game` ([`make_for`]), for the server to run on; `None` when none was asked for, the
/// error when it could not be made. Given back with [`give_back`].
pub(crate) fn take(game: vk::Device) -> Option<Result<Offload, String>> {
    MADE.lock().unwrap_or_else(std::sync::PoisonError::into_inner).iter_mut().find(|m| m.game == game)?.offload.take()
}

/// The server is done with the device (`busy`: a frame on it may still run, so it is never destroyed).
pub(crate) fn give_back(game: vk::Device, offload: Offload, busy: bool) {
    let mut all = MADE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    match all.iter_mut().find(|m| m.game == game) {
        Some(m) => {
            m.offload = Some(Ok(offload));
            m.busy = busy;
        }
        None => all.push(Made { game, offload: Some(Ok(offload)), busy }),
    }
}

/// Called inside the game's `vkDestroyDevice` for `game` (the loader's lock held), once its server was joined:
/// destroys the offload device made with it, unless a frame on it may still run (left to the process's end, logged).
///
/// # Safety
/// Inside `vkDestroyDevice` for `game`, after the server's worker was joined.
pub(crate) unsafe fn destroy_for(game: vk::Device) {
    let made = {
        let mut all = MADE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let k = all.iter().position(|m| m.game == game);
        k.map(|k| all.remove(k))
    };
    let Some(m) = made else { return };
    match m.offload {
        Some(Ok(off)) if !m.busy => {
            // SAFETY: nothing of it is alive (the server ended with it clean) and the loader's lock is held.
            unsafe { off.destroy() };
        }
        Some(Ok(off)) => {
            crate::log!("[offload] the other GPU's device is left to the process's end: a frame on it never finished");
            std::mem::forget(off);
        }
        _ => {}
    }
}

/// The offload's counters for the periodic summary: the server's frame on the other GPU (copies and network), and
/// the present's view (round trip, frames shown with an answer or without).
#[derive(Default)]
pub(crate) struct Stats {
    /// The server's frames: count, and the sum of their times (µs) on the other GPU.
    pub server_frames: AtomicU64,
    pub server_us: AtomicU64,
    /// Of that: the stand-in's copies, in to the other GPU and back (µs; the stand-in only).
    pub copy_in_us: AtomicU64,
    pub copy_out_us: AtomicU64,
    /// The presents: with the previous frame's answer composed, and without (it was not back, or none was asked).
    pub shown: AtomicU64,
    pub fell_back: AtomicU64,
    /// The time from a request to its answer, seen at the present (µs).
    pub round_trip_us: AtomicU64,
    pub round_trips: AtomicU64,
}

pub(crate) static STATS: Stats = Stats {
    server_frames: AtomicU64::new(0),
    server_us: AtomicU64::new(0),
    copy_in_us: AtomicU64::new(0),
    copy_out_us: AtomicU64::new(0),
    shown: AtomicU64::new(0),
    fell_back: AtomicU64::new(0),
    round_trip_us: AtomicU64::new(0),
    round_trips: AtomicU64::new(0),
};

/// Counts the delayed present's frames and logs [`Stats::summary`] every 600th.
pub(crate) fn note_present() {
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed) + 1;
    if n.is_multiple_of(600) {
        crate::log!("[offload] {}", STATS.summary());
        crate::logging::flush();
    }
}

impl Stats {
    /// One line for the log: the counters since the last line, then reset. The network's time on the other GPU is what
    /// the game's GPU no longer spends (with the copies in and out of system memory included there).
    pub(crate) fn summary(&self) -> String {
        let take = |a: &AtomicU64| a.swap(0, Ordering::Relaxed);
        let (frames, us, cin, cout) = (take(&self.server_frames), take(&self.server_us), take(&self.copy_in_us), take(&self.copy_out_us));
        let (shown, fell, rt, rts) = (take(&self.shown), take(&self.fell_back), take(&self.round_trip_us), take(&self.round_trips));
        let mean = |sum: u64, n: u64| if n == 0 { 0.0 } else { sum as f64 / n as f64 / 1000.0 };
        let saved = if cin + cout > 0 {
            "the stand-in copy ran in place of the network (NEURAL_FORGE_OFFLOAD_STANDIN)".to_string()
        } else {
            format!("the game's GPU saved about {:.2} ms a frame (the network ran there before)", mean(us, frames))
        };
        format!(
            "{frames} frames on the other GPU, {:.2} ms each there{}; request to answer {:.2} ms; presents {shown} with the answer, {fell} without; {saved}",
            mean(us, frames),
            if cin + cout > 0 { format!(" (copy in {:.2} ms, copy back {:.2} ms)", mean(cin, frames), mean(cout, frames)) } else { String::new() },
            mean(rt, rts),
        )
    }
}

/// The stand-in for the network on the other device ([`standin`]): the frame copied in to a buffer of the device and
/// back out into the answer, through imported host memory where the device allows, else staged. Its copies are timed.
pub(crate) struct Standin {
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    queries: vk::QueryPool,
    period_ns: f32,
    sized: Option<StandinBuffers>,
    /// `NEURAL_FORGE_OFFLOAD_TEST_STALL=<frame>` (tests of the stall path only): that frame's copy waits on the GPU for
    /// an event the host sets 8 s later, past the server's fence wait, so the frame stalls and then finishes.
    test_stall: Option<(u64, vk::Event)>,
    frames: u64,
}

struct StandinBuffers {
    bytes: u64,
    middle: (vk::Buffer, vk::DeviceMemory),
    /// The proxy and answer regions, imported where the device allows (`super::region_buffer`), else buffers of their
    /// own that the frame is staged through (`staged`).
    proxy: super::HostBuffer,
    answer: super::HostBuffer,
}

impl Standin {
    /// # Safety
    /// `device` is live; `queue_family` is the family of the queue it submits to.
    pub(crate) unsafe fn new(device: &ash::Device, instance: &ash::Instance, physical: vk::PhysicalDevice, queue_family: u32) -> Option<Self> {
        // SAFETY: valid create infos; everything made is destroyed by `destroy` (or here on failure).
        unsafe {
            let pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default().queue_family_index(queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .ok()?;
            let Ok(cmds) = crate::loader_data::allocate_commands(device, &vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(1))
            else {
                device.destroy_command_pool(pool, None);
                return None;
            };
            let Ok(fence) = device.create_fence(&vk::FenceCreateInfo::default(), None) else {
                device.destroy_command_pool(pool, None);
                return None;
            };
            // Timed only where the queue family writes timestamps (`timestampValidBits` 0: none).
            let timed = instance.get_physical_device_queue_family_properties(physical).get(queue_family as usize).is_some_and(|f| f.timestamp_valid_bits > 0);
            let queries = if timed {
                device.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(3), None).unwrap_or_default()
            } else {
                vk::QueryPool::null()
            };
            let period_ns = instance.get_physical_device_properties(physical).limits.timestamp_period;
            let test_stall = neural_forge_protocol::env::var("NEURAL_FORGE_OFFLOAD_TEST_STALL")
                .and_then(|v| v.parse::<u64>().ok())
                .and_then(|n| device.create_event(&vk::EventCreateInfo::default(), None).ok().map(|e| (n, e)));
            Some(Self { pool, cmd: cmds[0], fence, queries, period_ns, sized: None, test_stall, frames: 0 })
        }
    }

    /// The fence of its copy (the one a stall waits for).
    pub(crate) fn fence(&self) -> vk::Fence {
        self.fence
    }

    /// Copies the frame (`bytes` at `proxy`) in and back out into `answer` on `queue`, waited for (bounded). The two
    /// copies' times, or `None` (a failure, or a wait that timed out: `stalled`).
    ///
    /// # Safety
    /// The regions are mapped for the process's life with room for `bytes` (and aligned for an import, which is checked);
    /// nothing of this stand-in is pending.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn run(
        &mut self,
        device: &ash::Device,
        instance: &ash::Instance,
        physical: vk::PhysicalDevice,
        queue: vk::Queue,
        import: bool,
        regions: (*mut u8, *mut u8, usize),
        bytes: u64,
        stalled: &mut bool,
    ) -> Option<(Duration, Duration)> {
        let (proxy, answer, capacity) = regions;
        if self.sized.as_ref().is_some_and(|s| s.bytes != bytes) {
            // SAFETY: nothing pending (contract).
            unsafe { self.free(device) };
        }
        if self.sized.is_none() {
            // SAFETY: as the contract.
            self.sized = unsafe { Self::buffers(device, instance, physical, import, proxy, answer, capacity, bytes) };
        }
        let s = self.sized.as_ref()?;
        if s.proxy.staged {
            // SAFETY: host-coherent and mapped with room for `bytes`; the region holds the frame.
            unsafe { std::ptr::copy_nonoverlapping(proxy, s.proxy.ptr, bytes as usize) };
        }
        let (src, dst) = (s.proxy.buffer, s.answer.buffer);
        let region = [vk::BufferCopy::default().size(bytes)];
        let barrier = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ);
        let host = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
        // SAFETY: the command buffer is idle (its fence waited), every handle live.
        let recorded = unsafe {
            device.reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty()).is_ok()
                && device.begin_command_buffer(self.cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).is_ok()
                && {
                    let timed = self.queries != vk::QueryPool::null();
                    if timed {
                        device.cmd_reset_query_pool(self.cmd, self.queries, 0, 3);
                        device.cmd_write_timestamp(self.cmd, vk::PipelineStageFlags::TOP_OF_PIPE, self.queries, 0);
                    }
                    self.frames += 1;
                    if let Some((_, event)) = self.test_stall.filter(|(n, _)| *n == self.frames) {
                        crate::log!("[offload] NEURAL_FORGE_OFFLOAD_TEST_STALL: frame {} waits 8 s on the other GPU", self.frames);
                        set_later(device.handle(), device.fp_v1_0().set_event, event);
                        let host = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::HOST_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ);
                        device.cmd_wait_events(self.cmd, &[event], vk::PipelineStageFlags::HOST, vk::PipelineStageFlags::TRANSFER, &[host], &[], &[]);
                    }
                    device.cmd_copy_buffer(self.cmd, src, s.middle.0, &region);
                    device.cmd_pipeline_barrier(
                        self.cmd,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[barrier],
                        &[],
                        &[],
                    );
                    if timed {
                        device.cmd_write_timestamp(self.cmd, vk::PipelineStageFlags::TRANSFER, self.queries, 1);
                    }
                    device.cmd_copy_buffer(self.cmd, s.middle.0, dst, &region);
                    device.cmd_pipeline_barrier(
                        self.cmd,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::HOST,
                        vk::DependencyFlags::empty(),
                        &[host],
                        &[],
                        &[],
                    );
                    if timed {
                        device.cmd_write_timestamp(self.cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, self.queries, 2);
                    }
                    device.end_command_buffer(self.cmd).is_ok()
                }
        };
        if !recorded {
            return None;
        }
        let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&self.cmd));
        // SAFETY: the fence is idle; the queue is this thread's alone.
        let submitted = unsafe { device.reset_fences(&[self.fence]).and_then(|()| device.queue_submit(queue, &[submit], self.fence)) };
        if crate::note_vk(device.handle(), submitted).is_err() {
            return None;
        }
        // SAFETY: the fence was just submitted.
        let waited = unsafe { device.wait_for_fences(&[self.fence], true, crate::FENCE_WAIT_TIMEOUT.as_nanos() as u64) };
        if crate::note_fence_wait(device.handle(), waited, "offload::standin").is_err() {
            *stalled = true;
            return None;
        }
        // `NEURAL_FORGE_OFFLOAD_TEST_SLOW_MS` (tests of the late rule only): a second GPU this much slower per frame.
        static SLOW: std::sync::LazyLock<Option<u64>> =
            std::sync::LazyLock::new(|| neural_forge_protocol::env::var("NEURAL_FORGE_OFFLOAD_TEST_SLOW_MS").and_then(|v| v.parse().ok()));
        if let Some(ms) = *SLOW {
            std::thread::sleep(Duration::from_millis(ms));
        }
        if s.answer.staged {
            // SAFETY: the copy into it completed; both hold `bytes`.
            unsafe { std::ptr::copy_nonoverlapping(s.answer.ptr, answer, bytes as usize) };
        }
        let mut stamps = [0u64; 3];
        // SAFETY: the three queries were written by the completed buffer.
        let read = self.queries != vk::QueryPool::null()
            && unsafe { device.get_query_pool_results(self.queries, 0, &mut stamps, vk::QueryResultFlags::TYPE_64) }.is_ok();
        let ns = |a: u64, b: u64| Duration::from_nanos((b.saturating_sub(a) as f64 * f64::from(self.period_ns)) as u64);
        Some(if read { (ns(stamps[0], stamps[1]), ns(stamps[1], stamps[2])) } else { (Duration::ZERO, Duration::ZERO) })
    }

    /// # Safety
    /// As [`Self::run`].
    #[allow(clippy::too_many_arguments)]
    unsafe fn buffers(
        device: &ash::Device,
        instance: &ash::Instance,
        physical: vk::PhysicalDevice,
        import: bool,
        proxy: *mut u8,
        answer: *mut u8,
        capacity: usize,
        bytes: u64,
    ) -> Option<StandinBuffers> {
        // SAFETY: plain property query.
        let props = unsafe { instance.get_physical_device_memory_properties(physical) };
        let info = vk::BufferCreateInfo::default().size(bytes).usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST);
        // SAFETY: valid create info; destroyed on every failure below.
        let middle = unsafe {
            let buffer = device.create_buffer(&info, None).ok()?;
            let reqs = device.get_buffer_memory_requirements(buffer);
            let Some(t) = (0..props.memory_type_count).find(|&i| {
                reqs.memory_type_bits & (1 << i) != 0 && props.memory_types[i as usize].property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            }) else {
                device.destroy_buffer(buffer, None);
                return None;
            };
            let Ok(memory) = device.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(reqs.size).memory_type_index(t), None) else {
                device.destroy_buffer(buffer, None);
                return None;
            };
            if device.bind_buffer_memory(buffer, memory, 0).is_err() {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
                return None;
            }
            (buffer, memory)
        };
        let usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        // SAFETY: the regions are mapped for the process's life with `capacity` bytes each (the contract).
        let proxy_buffer = unsafe { super::region_buffer(device, instance, physical, import, proxy, capacity, bytes, usage) };
        let answer_buffer = unsafe { super::region_buffer(device, instance, physical, import, answer, capacity, bytes, usage) };
        let (proxy_buffer, answer_buffer) = match (proxy_buffer, answer_buffer) {
            (Some(p), Some(a)) => (p, a),
            (p, a) => {
                // SAFETY: none of it was used.
                unsafe {
                    for b in [p, a].into_iter().flatten() {
                        b.destroy(device);
                    }
                    device.destroy_buffer(middle.0, None);
                    device.free_memory(middle.1, None);
                }
                return None;
            }
        };
        crate::log!(
            "[offload] stand-in copy for {bytes} bytes: {}",
            if proxy_buffer.staged || answer_buffer.staged { "staged through host-visible buffers" } else { "the shared regions imported as host memory" }
        );
        Some(StandinBuffers { bytes, middle, proxy: proxy_buffer, answer: answer_buffer })
    }

    /// # Safety
    /// Nothing of the stand-in is pending.
    unsafe fn free(&mut self, device: &ash::Device) {
        if let Some(s) = self.sized.take() {
            // SAFETY: as the contract.
            unsafe {
                device.destroy_buffer(s.middle.0, None);
                device.free_memory(s.middle.1, None);
                s.proxy.destroy(device);
                s.answer.destroy(device);
            }
        }
    }

    /// # Safety
    /// Nothing of the stand-in is pending.
    pub(crate) unsafe fn destroy(mut self, device: &ash::Device) {
        // SAFETY: as the contract.
        unsafe {
            self.free(device);
            device.destroy_query_pool(self.queries, None);
            if let Some((_, event)) = self.test_stall {
                device.destroy_event(event, None);
            }
            device.destroy_fence(self.fence, None);
            device.destroy_command_pool(self.pool, None);
        }
    }
}

/// Sets `event` from another thread 8 s from now (`NEURAL_FORGE_OFFLOAD_TEST_STALL`).
fn set_later(device: vk::Device, set: vk::PFN_vkSetEvent, event: vk::Event) {
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_secs(8));
        // SAFETY: the event outlives the wait (destroyed with the stand-in, once its frame finished).
        let _ = unsafe { set(device, event) };
    });
}

/// The other GPU's frames as the server sees them: one over [`HUNG`] after the first [`WARM_FRAMES`] of a build, or
/// none run for [`NOT_RUNNING`] while frames come, falls back.
#[derive(Default)]
pub(crate) struct ServerRule {
    /// The build the frames ran with, and how many ran with it.
    generation: u64,
    warm: u32,
    not_running_since: Option<std::time::Instant>,
}

impl ServerRule {
    /// Notes a frame the network ran on (`generation`: the build's), and how long it took; why to fall back.
    pub(crate) fn ran(&mut self, generation: u64, took: Duration) -> Option<String> {
        self.not_running_since = None;
        if generation != self.generation {
            self.generation = generation;
            self.warm = 0;
        }
        self.warm = self.warm.saturating_add(1);
        (self.warm > WARM_FRAMES && took >= HUNG).then(|| format!("a frame took {took:.0?} on the other GPU"))
    }

    /// Notes a frame the model was wanted for that the network did not run on (`why`: the loader's reason, if any), at
    /// `now`; why to fall back.
    pub(crate) fn missed(&mut self, now: std::time::Instant, why: Option<&str>) -> Option<String> {
        let since = *self.not_running_since.get_or_insert(now);
        (now.saturating_duration_since(since) >= NOT_RUNNING)
            .then(|| format!("the network has not run on the other GPU for {NOT_RUNNING:?}{}", why.map_or(String::new(), |w| format!(" ({w})"))))
    }
}

/// The other GPU's frames as the present sees them: more than [`LATE_ALLOWED`] of the last [`LATE_WINDOW`] sent frames
/// late ([`LateRule::is_late`]), or [`UNANSWERED_IN_A_ROW`] in a row with no answer at all, falls back, decided as soon
/// as that is so. Judged where the game waits, so a second GPU far too slow for the game is caught whatever its own
/// times are.
#[derive(Default)]
pub(crate) struct LateRule {
    window: std::collections::VecDeque<bool>,
    unanswered: u32,
    /// The frame size judged ([`Self::note`]), and how many frames were noted at it: the first [`WARM_FRAMES`] after
    /// a start or a change of size (the network built for it) count for nothing.
    dims: Option<(u32, u32)>,
    seen: u32,
}

impl LateRule {
    /// Whether a frame is late: `answer` its request-to-answer time (`None`: none came in the budget), `interval` the time
    /// between its present and the next.
    pub(crate) fn is_late(answer: Option<Duration>, interval: Duration) -> bool {
        answer.is_none_or(|a| a > (interval * LATE_INTERVALS).max(LATE_FLOOR))
    }

    /// Notes a sent frame (`answer` as [`Self::is_late`] takes it); why to fall back.
    pub(crate) fn note(&mut self, answer: Option<Duration>, interval: Duration, dims: (u32, u32)) -> Option<String> {
        if self.dims != Some(dims) {
            *self = Self { dims: Some(dims), ..Self::default() };
        }
        self.seen = self.seen.saturating_add(1);
        if self.seen <= WARM_FRAMES {
            return None;
        }
        self.unanswered = if answer.is_none() { self.unanswered + 1 } else { 0 };
        if self.unanswered >= UNANSWERED_IN_A_ROW {
            return Some(format!("{UNANSWERED_IN_A_ROW} frames in a row got no answer from the other GPU within the present's wait"));
        }
        if self.window.len() == LATE_WINDOW {
            self.window.pop_front();
        }
        self.window.push_back(Self::is_late(answer, interval));
        let n = self.window.iter().filter(|&&l| l).count();
        (n > LATE_ALLOWED).then(|| {
            format!(
                "{n} of the last {} frames came back from the other GPU later than {LATE_INTERVALS} frame intervals and {LATE_FLOOR:?}: it is too slow for this game",
                self.window.len()
            )
        })
    }
}

/// The present's verdict on the other GPU ([`LateRule`]), for the server to end on.
static GIVE_UP: Mutex<Option<String>> = Mutex::new(None);

/// The present gives the other GPU up (`why`); the server ends at its next turn.
pub(crate) fn give_up(why: String) {
    GIVE_UP.lock().unwrap_or_else(std::sync::PoisonError::into_inner).get_or_insert(why);
}

/// Takes the present's verdict ([`give_up`]).
pub(crate) fn take_give_up() -> Option<String> {
    GIVE_UP.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take()
}

/// The present side's rule (one per process: one swapchain is the primary).
pub(crate) static LATE: Mutex<Option<LateRule>> = Mutex::new(None);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_choice_parses_every_form_and_refuses_the_rest() {
        assert_eq!(Choice::parse(""), Ok(Choice::Off));
        assert_eq!(Choice::parse(" OFF "), Ok(Choice::Off));
        assert_eq!(Choice::parse("same"), Ok(Choice::Same));
        assert_eq!(Choice::parse("1"), Ok(Choice::Index(1)));
        assert_eq!(Choice::parse("10de:2c05"), Ok(Choice::Pci { vendor: 0x10de, device: 0x2c05, bus: None }));
        let bus = PciAddress { domain: 0, bus: 0x73, device: 0, function: 0 };
        assert_eq!(Choice::parse("10DE:2C05@0000:73:00.0"), Ok(Choice::Pci { vendor: 0x10de, device: 0x2c05, bus: Some(bus) }));
        assert_eq!(Choice::parse("10de:2c05@73:00.0"), Ok(Choice::Pci { vendor: 0x10de, device: 0x2c05, bus: Some(bus) }));
        assert!(Choice::parse("nvidia").is_err());
        assert_eq!(Choice::parse("0"), Ok(Choice::Index(0)));
        assert!(Choice::parse("0x0").is_err(), "no hexadecimal index");
        assert!(Choice::parse("+1").is_err());
        assert!(Choice::parse("-1").is_err());
        assert!(Choice::parse("0x10de:2c05").is_err(), "IDs without 0x");
        assert!(Choice::parse("10de0:2c05").is_err(), "IDs of 4 digits at most");
        assert!(Choice::parse("10de:2c05@0000:73:20.0").is_err(), "a device number above 0x1f");
        assert!(Choice::parse("10de:2c05@0000:73:00.8").is_err(), "a function above 7");
        assert!(Choice::parse("10de:2c05@00000:73:00.0").is_err(), "a domain of 5 digits");
        assert!(Choice::parse("10de:2c05@+7:00.0").is_err());
        assert!(Choice::parse("10de:zz").is_err());
        assert!(Choice::parse("10de:2c05@73").is_err());
        assert_eq!(bus.to_string(), "0000:73:00.0");
    }

    fn gpu(index: u32, vendor: u32, device: u32, bus: u32, game: bool) -> Candidate {
        Candidate { index, vendor, device, pci: Some(PciAddress { domain: 0, bus, device: 0, function: 0 }), game }
    }

    #[test]
    fn the_choice_picks_its_gpu_and_never_the_games_unless_asked() {
        // A 4090 the game runs on, a 5080 and an iGPU.
        let all = [gpu(0, 0x10de, 0x2684, 1, true), gpu(1, 0x10de, 0x2c02, 0x73, false), gpu(2, 0x8086, 0x4680, 0, false)];
        assert_eq!(pick(&Choice::Same, &all), Ok(0));
        assert_eq!(pick(&Choice::Index(1), &all), Ok(1));
        assert_eq!(pick(&Choice::Pci { vendor: 0x8086, device: 0x4680, bus: None }, &all), Ok(2));
        assert!(pick(&Choice::Index(0), &all).unwrap_err().contains("game's own"));
        assert!(pick(&Choice::Index(7), &all).is_err());
        assert!(pick(&Choice::Off, &all).is_err());
        // Two of the same card: the one the game is not on, or by its address.
        let twins = [gpu(0, 0x10de, 0x2c02, 1, true), gpu(1, 0x10de, 0x2c02, 2, false)];
        assert_eq!(pick(&Choice::Pci { vendor: 0x10de, device: 0x2c02, bus: None }, &twins), Ok(1));
        let three = [gpu(0, 0x10de, 0x2684, 1, true), gpu(1, 0x10de, 0x2c02, 2, false), gpu(2, 0x10de, 0x2c02, 3, false)];
        assert!(pick(&Choice::Pci { vendor: 0x10de, device: 0x2c02, bus: None }, &three).unwrap_err().contains("PCI address"));
        let at3 = Some(PciAddress { domain: 0, bus: 3, device: 0, function: 0 });
        assert_eq!(pick(&Choice::Pci { vendor: 0x10de, device: 0x2c02, bus: at3 }, &three), Ok(2));
    }

    /// Given up is for the process and the session: nothing later reads as running again.
    #[test]
    fn a_fall_back_is_for_the_whole_process() {
        assert!(!given_up());
        note_state(State::Active("GPU".into()));
        assert_eq!(status_line().as_deref(), Some("second GPU: the model runs on GPU, each frame shown one frame later"));
        note_state(State::Stalled("GPU".into()));
        assert!(status_line().is_some_and(|l| l.contains("stalled")), "a stall is not shown as running");
        note_state(State::FellBack("too slow".into()));
        assert!(given_up());
        note_state(State::Starting("GPU".into()));
        note_state(State::Active("GPU".into()));
        assert!(given_up(), "another device's server cannot start it again");
        assert!(status_line().is_some_and(|l| l.contains("given up (too slow)")));
        *STATE.lock().unwrap() = State::Off;
    }

    #[test]
    fn the_queue_family_prefers_compute_without_graphics() {
        let f = |flags: vk::QueueFlags, count: u32| vk::QueueFamilyProperties { queue_flags: flags, queue_count: count, ..Default::default() };
        let nvidia = [f(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE, 16), f(vk::QueueFlags::TRANSFER, 2), f(vk::QueueFlags::COMPUTE, 8)];
        assert_eq!(family(&nvidia, 2), Some(2));
        let igpu = [f(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE, 1)];
        assert_eq!(family(&igpu, 1), Some(0));
        assert_eq!(family(&igpu, 2), None);
    }

    #[test]
    fn a_hung_frame_falls_back_except_while_a_build_warms_up() {
        let mut rule = ServerRule::default();
        for _ in 0..WARM_FRAMES {
            assert!(rule.ran(1, HUNG * 3).is_none(), "the first frames of a build may be slow");
        }
        assert!(rule.ran(1, Duration::from_millis(20)).is_none());
        assert!(rule.ran(1, HUNG).is_some());
        let mut rule = ServerRule::default();
        for _ in 0..WARM_FRAMES + 1 {
            assert!(rule.ran(1, Duration::from_millis(20)).is_none());
        }
        assert!(rule.ran(2, HUNG * 2).is_none(), "a new build warms up again");
    }

    #[test]
    fn frames_the_network_never_runs_on_fall_back_after_a_while() {
        let mut rule = ServerRule::default();
        let t0 = std::time::Instant::now();
        assert!(rule.missed(t0, None).is_none());
        assert!(rule.missed(t0 + NOT_RUNNING / 2, Some("the network could not be built")).is_none());
        let why = rule.missed(t0 + NOT_RUNNING, Some("paused: too little VRAM")).expect("gives up");
        assert!(why.contains("too little VRAM"));
        let mut rule = ServerRule::default();
        assert!(rule.missed(t0, None).is_none());
        assert!(rule.ran(1, Duration::from_millis(5)).is_none());
        assert!(rule.missed(t0 + NOT_RUNNING, None).is_none(), "a frame that ran starts the count over");
    }

    #[test]
    fn late_frames_fall_back_only_when_far_past_the_frame_and_the_floor() {
        let ms = Duration::from_millis;
        assert!(!LateRule::is_late(Some(ms(25)), ms(10)));
        assert!(!LateRule::is_late(Some(ms(35)), ms(10)), "past 3 intervals but under the floor");
        assert!(LateRule::is_late(Some(ms(120)), ms(10)));
        assert!(!LateRule::is_late(Some(ms(250)), ms(100)), "a slow game: 3 intervals");
        assert!(LateRule::is_late(None, ms(10)));
        const D: (u32, u32) = (64, 64);
        // The first frames after a start or a change of size count for nothing (the network is built then).
        let mut rule = LateRule::default();
        assert!((0..WARM_FRAMES).all(|_| rule.note(None, ms(16), D).is_none()));
        assert!((0..WARM_FRAMES).all(|_| rule.note(None, ms(16), (32, 32)).is_none()), "a new size warms up again");
        let mut rule = LateRule::default();
        for _ in 0..WARM_FRAMES {
            rule.note(Some(ms(5)), ms(16), D);
        }
        // A second GPU at 30 ms a frame against a 7 ms game (frame generation at ~144 fps) beats the network after the
        // game's frame on one GPU: it must not fall back, however long it runs.
        let mut rule = LateRule::default();
        assert!((0..LATE_WINDOW * 3).all(|_| rule.note(Some(ms(30)), ms(7), D).is_none()));
        // A GPU far too slow (150 ms against a 16 ms game) is given up as soon as more than the allowance came late.
        let mut rule = LateRule::default();
        let at = (0..LATE_WINDOW).position(|_| rule.note(Some(ms(150)), ms(16), D).is_some());
        assert_eq!(at, Some(LATE_ALLOWED + WARM_FRAMES as usize), "decided at the first frame over the allowance, not after a full window");
        // A few late frames (a hitch) do not.
        let mut rule = LateRule::default();
        assert!((0..LATE_WINDOW * 3).all(|k| rule.note(Some(if k % 10 == 0 { ms(150) } else { ms(20) }), ms(16), D).is_none()));
        // No answer at all: a few in a row are enough.
        let mut rule = LateRule::default();
        let at = (0..LATE_WINDOW).position(|_| rule.note(None, ms(16), D).is_some());
        assert_eq!(at, Some(WARM_FRAMES as usize + UNANSWERED_IN_A_ROW as usize - 1));
        let mut rule = LateRule::default();
        assert!((0..8).all(|k| rule.note(if k % 4 == 0 { Some(ms(20)) } else { None }, ms(16), D).is_none()), "an answer starts the count over");
    }
}
