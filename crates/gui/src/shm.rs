//! Binds GTK widgets to `neural_forge_protocol::ShmHeader` fields — this crate's equivalent
//! of upstream's `shm_binder.h/.cpp`, written fresh against the protocol crate's Rust
//! types (there's no logic in a per-field binder worth porting either way, just a
//! mechanical widget<->field mapping).
//!
//! Also persists every [`neural_forge_protocol::ShmHeader::persisted_settings`] value to
//! `config.ini` on change, and applies whatever was last persisted when this call is
//! the one that created the mapping fresh (see `Shm::open`) -- the SHM mapping itself
//! lives under `/tmp` and does not survive a reboot, so without this, every tuning
//! change would quietly reset on the next login. Confirmed missing by comparing
//! against upstream DLSS5VKLayer's real, installed `config.ini` on 2026-09-10, which
//! persists the equivalent of every one of these fields.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use neural_forge_protocol::mapping::Mapping;

/// How a bound field's raw bits are to be read back.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FieldKind {
    Float,
    Int,
}

/// A way to read a bound field's current raw bits, so a widget can follow changes made by
/// something other than itself (`shmctl`, loading a profile, Reset, another GUI).
pub type Reader = Box<dyn Fn() -> u32>;

thread_local! {
    /// The reader of the most recent `bind_*` call. Every row builder in `ui.rs` is called
    /// immediately after the binding it displays and takes it from here.
    static PENDING_READER: std::cell::RefCell<Option<(FieldKind, Reader, Option<&'static str>)>> = const { std::cell::RefCell::new(None) };
    /// The setting name that came with the reader [`take_pending_reader`] last handed out, for the row's
    /// per-game mark ([`take_pending_name`]).
    static TAKEN_NAME: std::cell::Cell<Option<&'static str>> = const { std::cell::Cell::new(None) };
}

/// Takes the reader the last `bind_*` call left for the row being built now.
pub fn take_pending_reader() -> Option<(FieldKind, Reader)> {
    let (kind, read, name) = PENDING_READER.with(|p| p.borrow_mut().take())?;
    TAKEN_NAME.with(|t| t.set(name));
    Some((kind, read))
}

/// The persisted setting the row just built displays (the name its `bind_*` call was given), once.
pub fn take_pending_name() -> Option<&'static str> {
    TAKEN_NAME.with(std::cell::Cell::take)
}

fn leave_reader(
    shm: &Arc<Mapping>,
    kind: FieldKind,
    name: Option<&'static str>,
    get: impl Fn(&neural_forge_protocol::ShmHeader) -> &std::sync::atomic::AtomicU32 + 'static,
) -> impl Fn(&neural_forge_protocol::ShmHeader) -> &std::sync::atomic::AtomicU32 + 'static {
    let get = std::rc::Rc::new(get);
    let reader_get = std::rc::Rc::clone(&get);
    let reader_shm = Arc::clone(shm);
    PENDING_READER.with(|p| {
        *p.borrow_mut() = Some((kind, Box::new(move || reader_get(reader_shm.header()).load(Ordering::Relaxed)), name));
    });
    move |h| get(h)
}

/// Wraps the open mapping so the UI module can pass one `Arc` around to every
/// callback instead of re-opening or re-threading raw pointers everywhere.
pub struct Shm(pub Arc<Mapping>);

impl Shm {
    /// Opens the channel `config.ini` names (see `neural_forge_supervisor::channel_path`).
    /// A header laid out by another build is never handed out or rewritten: that comes
    /// back as `OpenError::WrongVersion` for the caller to show.
    pub fn open() -> Result<Self, neural_forge_protocol::mapping::OpenError> {
        let cfg = neural_forge_supervisor::Config::load();
        let mapping = neural_forge_supervisor::open_channel(&cfg)?;
        neural_forge_supervisor::apply_saved_settings(&cfg, &mapping);
        Ok(Shm(Arc::new(mapping)))
    }
}

/// How long after the last change a setting is written to `config.ini`: a spin button held down
/// or dragged changes the value many times a second, and each change was a full rewrite.
const SAVE_DELAY: std::time::Duration = std::time::Duration::from_millis(300);

use neural_forge_supervisor::game_settings::LiveScope;

#[derive(Default)]
struct PendingSave {
    /// Settings changed since the last save, by whose settings the header held when each was changed (the
    /// scope it is saved to), with the value it was changed to. While the header still holds that scope the
    /// value saved is the header's own at the time of the save, so a Reset or profile load in between is never
    /// overwritten with an older value; once it holds another's (a game's session started, `game-settings`
    /// switched it), the value recorded here is saved to the scope it was made in.
    changes: std::collections::BTreeMap<(LiveScope, &'static str), String>,
    mapping: Option<Arc<Mapping>>,
    timer: Option<glib::SourceId>,
    on_error: Option<std::rc::Rc<dyn Fn(String)>>,
}

thread_local! {
    static PENDING_SAVE: std::cell::RefCell<PendingSave> = std::cell::RefCell::new(PendingSave::default());
}

/// Where a failed save is reported (the window's toasts).
pub fn on_save_error(f: impl Fn(String) + 'static) {
    PENDING_SAVE.with(|p| p.borrow_mut().on_error = Some(std::rc::Rc::new(f)));
}

/// Marks `name` changed and (re)starts the save timer, so a burst of changes is one write.
fn persist_later(shm: &Arc<Mapping>, name: &'static str) {
    let header = shm.header();
    let Some(value) = neural_forge_protocol::persist::snapshot(header).remove(&format!("set_{name}")) else { return };
    let scope = LiveScope::of(header);
    PENDING_SAVE.with(|p| {
        let mut p = p.borrow_mut();
        p.changes.insert((scope, name), value);
        p.mapping = Some(Arc::clone(shm));
        if let Some(timer) = p.timer.take() {
            timer.remove();
        }
        p.timer = Some(glib::timeout_add_local_once(SAVE_DELAY, || {
            PENDING_SAVE.with(|p| p.borrow_mut().timer = None);
            if let Err(e) = flush_settings() {
                let on_error = PENDING_SAVE.with(|p| p.borrow().on_error.clone());
                match on_error {
                    Some(report) => report(format!("Saving the settings failed: {e}")),
                    None => eprintln!("neural-forge: saving the settings failed: {e}"),
                }
            }
        }));
    });
}

/// Marks `names` changed for the next save: settings written to the header some other way than
/// through a bound row (a preset, `Knobs::write`).
pub fn persist_settings_later(shm: &Arc<Mapping>, names: &[&'static str]) {
    for name in names {
        persist_later(shm, name);
    }
}

/// Writes every changed setting now (the timer does this on its own; the window calls it on close, and the
/// "Settings for" menu before it changes whose settings the header holds), each to the settings it was
/// changed in: `config.ini` for all games, else that game's own (`game_settings::save_values`). A full
/// load-modify-save of each file, so no other key is written stale.
pub fn flush_settings() -> std::io::Result<()> {
    let (changes, mapping) = PENDING_SAVE.with(|p| {
        let mut p = p.borrow_mut();
        if let Some(timer) = p.timer.take() {
            timer.remove();
        }
        (std::mem::take(&mut p.changes), p.mapping.clone())
    });
    let Some(mapping) = mapping.filter(|_| !changes.is_empty()) else { return Ok(()) };
    let header = mapping.header();
    let held = header.settings_game();
    let now = neural_forge_protocol::persist::snapshot(header);
    let mut by_scope: std::collections::BTreeMap<LiveScope, std::collections::BTreeMap<String, String>> = std::collections::BTreeMap::new();
    for ((scope, name), value) in &changes {
        let key = format!("set_{name}");
        // Still the scope it was changed in: the header's value now (a Reset or a profile may have changed it).
        let value = if scope.scope == held { now.get(&key).unwrap_or(value) } else { value };
        by_scope.entry(scope.clone()).or_default().insert(key, value.clone());
    }
    let mut result = Ok(());
    for (scope, values) in by_scope {
        if let Err(e) = neural_forge_supervisor::game_settings::save_values(&scope, &values) {
            // Still unsaved: the next change, or the flush on close, writes them again (to the same scope).
            PENDING_SAVE.with(|p| {
                let mut p = p.borrow_mut();
                for ((s, name), value) in &changes {
                    if *s == scope {
                        p.changes.entry((s.clone(), *name)).or_insert_with(|| value.clone());
                    }
                }
            });
            if result.is_ok() {
                result = Err(e);
            }
        }
    }
    result
}

/// Binds a GTK `Scale`/`SpinButton`-shaped float control to one `f32`-bits field:
/// reads the current value to initialize the widget, and writes back (bumping
/// `control_seq` so the layer notices) whenever the widget changes. `name` must
/// match one of `ShmHeader::persisted_settings`'s names for the value to survive a
/// reboot; pass `None` for a field that isn't meant to (there are none of those among
/// the GUI's rows today, but the option exists for e.g. a future debug-only control).
pub fn bind_float(
    shm: &Arc<Mapping>,
    name: Option<&'static str>,
    get: impl Fn(&neural_forge_protocol::ShmHeader) -> &std::sync::atomic::AtomicU32 + 'static,
) -> (f32, impl Fn(f32) + 'static) {
    let get = leave_reader(shm, FieldKind::Float, name, get);
    let initial = f32::from_bits(get(shm.header()).load(Ordering::Relaxed));
    let shm = Arc::clone(shm);
    let setter = move |value: f32| {
        get(shm.header()).store(value.to_bits(), Ordering::Relaxed);
        shm.header().control_seq.fetch_add(1, Ordering::Relaxed);
        if let Some(name) = name {
            persist_later(&shm, name);
        }
    };
    (initial, setter)
}

/// Same shape as [`bind_float`], for a plain `u32`-valued field (enable flags,
/// enum-valued settings, etc).
pub fn bind_u32(
    shm: &Arc<Mapping>,
    name: Option<&'static str>,
    get: impl Fn(&neural_forge_protocol::ShmHeader) -> &std::sync::atomic::AtomicU32 + 'static,
) -> (u32, impl Fn(u32) + 'static) {
    let get = leave_reader(shm, FieldKind::Int, name, get);
    let initial = get(shm.header()).load(Ordering::Relaxed);
    let shm = Arc::clone(shm);
    let setter = move |value: u32| {
        get(shm.header()).store(value, Ordering::Relaxed);
        shm.header().control_seq.fetch_add(1, Ordering::Relaxed);
        if let Some(name) = name {
            persist_later(&shm, name);
        }
    };
    (initial, setter)
}

/// Same shape again, for a boolean flag stored as `0`/`1` in a `u32` field.
pub fn bind_bool(
    shm: &Arc<Mapping>,
    name: Option<&'static str>,
    get: impl Fn(&neural_forge_protocol::ShmHeader) -> &std::sync::atomic::AtomicU32 + 'static,
) -> (bool, impl Fn(bool) + 'static) {
    let (initial, setter) = bind_u32(shm, name, get);
    (initial != 0, move |value: bool| setter(if value { 1 } else { 0 }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tests set process-wide variables (the XDG dirs, the channel): one at a time.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Config, data and state in `scratch`, and the channel at `scratch/shm.bin`, until dropped.
    struct ScratchEnv {
        _guard: std::sync::MutexGuard<'static, ()>,
        prev: Vec<(&'static str, Option<String>)>,
        dir: std::path::PathBuf,
    }

    impl ScratchEnv {
        fn new(tag: &str) -> Self {
            let guard = ENV_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            let dir = std::env::temp_dir().join(format!("neural-forge-shm-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(dir.join("config/neural-forge")).unwrap();
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).unwrap();
            let vars = [
                ("XDG_CONFIG_HOME", dir.join("config")),
                ("XDG_DATA_HOME", dir.join("data")),
                ("XDG_STATE_HOME", dir.join("state")),
                ("NEURAL_FORGE_SHM", dir.join("shm.bin")),
            ];
            let prev = vars.iter().map(|(var, _)| (*var, std::env::var(var).ok())).collect();
            for (var, value) in &vars {
                std::env::set_var(var, value);
            }
            Self { _guard: guard, prev, dir }
        }

        fn read(&self, name: &str) -> String {
            std::fs::read_to_string(self.dir.join("config/neural-forge").join(name)).unwrap_or_default()
        }
    }

    impl Drop for ScratchEnv {
        fn drop(&mut self) {
            for (var, value) in self.prev.drain(..) {
                match value {
                    Some(v) => std::env::set_var(var, v),
                    None => std::env::remove_var(var),
                }
            }
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    /// A change still waiting to be saved goes to the settings it was made in, even when the header holds
    /// another's by the time it is saved (a game's session started, `game-settings` switched it), and while
    /// the header still holds them, the header's value is saved (a profile or Reset in between wins).
    #[test]
    fn a_pending_change_is_saved_to_the_scope_it_was_made_in() {
        use neural_forge_supervisor::game_settings::GameKey;
        let env = ScratchEnv::new("pending-scope");
        // A game with settings of its own.
        std::fs::write(env.dir.join("config/neural-forge/games.ini"), "[570]\nset_style=1\n").unwrap();
        let shm = Shm::open().unwrap();
        let header = shm.0.header();
        let (_, set_intensity) = bind_float(&shm.0, Some("intensity"), |h| &h.intensity_bits);

        // Changed for all games; the game's session starts (its own merge, from config.ini without the change)
        // before the save.
        set_intensity(1.25);
        neural_forge_protocol::persist::apply_scope(header, Some(&GameKey::Steam(570)));
        assert_eq!(header.settings_game(), 570);
        flush_settings().unwrap();
        assert!(env.read("config.ini").contains("set_intensity=1.25"), "all games' change is kept:\n{}", env.read("config.ini"));
        assert!(!env.read("games.ini").contains("set_intensity"), "and not made the game's:\n{}", env.read("games.ini"));

        // Changed for the game; the header goes back to all games' before the save.
        set_intensity(0.75);
        neural_forge_protocol::persist::apply_scope(header, None);
        flush_settings().unwrap();
        assert!(env.read("games.ini").contains("set_intensity=0.75"), "{}", env.read("games.ini"));
        assert!(env.read("config.ini").contains("set_intensity=1.25"), "{}", env.read("config.ini"));

        // The same scope, a profile loaded in between: the header's value, not the older change.
        set_intensity(2.0);
        header.write_settings(&[("intensity", 1.5f32.to_bits())]);
        flush_settings().unwrap();
        assert!(env.read("config.ini").contains("set_intensity=1.5"), "{}", env.read("config.ini"));
    }

    /// Exercises the real production path end to end: change a value through
    /// `bind_float`'s setter (as a GUI slider would), confirm it landed in
    /// `config.ini`, then open a brand-new mapping (simulating a reboot -- a fresh
    /// `/tmp`) and confirm `Shm::open` applied the persisted value automatically
    /// instead of leaving the hardcoded default.
    #[test]
    fn a_setting_changed_through_bind_float_survives_a_simulated_reboot() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let scratch = std::env::temp_dir().join(format!("neural-forge-shm-persist-test-{}", std::process::id()));
        let config_home = scratch.join("config");
        let shm_path = scratch.join("shm.bin");
        std::fs::create_dir_all(&config_home).unwrap();
        // The channel's directory must be private, as the real runtime directory is.
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&scratch, std::fs::Permissions::from_mode(0o700)).unwrap();

        // Config, data and state all go to the scratch dir, so the test never creates the real
        // XDG dirs.
        let vars = [
            ("XDG_CONFIG_HOME", config_home.clone()),
            ("XDG_DATA_HOME", scratch.join("data")),
            ("XDG_STATE_HOME", scratch.join("state")),
            ("NEURAL_FORGE_SHM", shm_path.clone()),
        ];
        let prev: Vec<(&str, Option<String>)> = vars.iter().map(|(var, _)| (*var, std::env::var(var).ok())).collect();
        for (var, value) in &vars {
            std::env::set_var(var, value);
        }

        {
            let shm = Shm::open().expect("first open should succeed and create a fresh mapping");
            let (initial, set_intensity) = bind_float(&shm.0, Some("intensity"), |h| &h.intensity_bits);
            assert_eq!(initial, 1.0, "default intensity should be the hardcoded default on a fresh mapping");
            set_intensity(1.5);
            set_intensity(1.75);
            // Debounced: nothing is written until the timer fires or the window flushes.
            let early = std::fs::read_to_string(config_home.join("neural-forge/config.ini")).unwrap_or_default();
            assert!(!early.contains("set_intensity"), "a change is saved after a pause, not on every step:\n{early}");
            // A save that fails (here: a directory where config.ini goes) keeps the change for the next one.
            let blocker = config_home.join("neural-forge/config.ini");
            std::fs::create_dir_all(&blocker).unwrap();
            assert!(flush_settings().is_err(), "config.ini cannot be written over a directory");
            std::fs::remove_dir(&blocker).unwrap();
            flush_settings().expect("save config.ini");
        }

        let saved = std::fs::read_to_string(config_home.join("neural-forge/config.ini")).expect("config.ini should exist");
        assert!(saved.contains("set_intensity=1.75"), "config.ini should have the new value:\n{saved}");

        // Simulate a reboot: the SHM file is what actually goes away (/tmp), not the
        // config file, so a fresh mapping at the same path is the right reproduction.
        std::fs::remove_file(&shm_path).ok();
        let shm = Shm::open().expect("second open should succeed and create another fresh mapping");
        let intensity_after = f32::from_bits(shm.0.header().intensity_bits.load(Ordering::Relaxed));
        assert_eq!(intensity_after, 1.75, "persisted value should have been applied on the fresh mapping");

        // The layer re-initialising the header (a game started first, or a version
        // change) must not lose the saved settings either: the next open puts them back.
        shm.0.header().init_defaults();
        assert_eq!(f32::from_bits(shm.0.header().intensity_bits.load(Ordering::Relaxed)), 1.0);
        drop(shm);
        let shm = Shm::open().expect("third open");
        assert_eq!(
            f32::from_bits(shm.0.header().intensity_bits.load(Ordering::Relaxed)),
            1.75,
            "a header someone else re-initialised must get the saved settings back"
        );
        // And only once: a value changed live afterwards is not overwritten by the next open.
        shm.0.header().intensity_bits.store(0.5f32.to_bits(), Ordering::Relaxed);
        drop(shm);
        let shm = Shm::open().expect("fourth open");
        assert_eq!(f32::from_bits(shm.0.header().intensity_bits.load(Ordering::Relaxed)), 0.5);
        drop(shm);

        for (var, value) in prev {
            match value {
                Some(v) => std::env::set_var(var, v),
                None => std::env::remove_var(var),
            }
        }
        std::fs::remove_dir_all(&scratch).ok();
    }
}
