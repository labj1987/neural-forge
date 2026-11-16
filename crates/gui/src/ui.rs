//! The window: the toolkit set-up, the pieces every page is built from (groups, rows, spin fields, menus, dialogs,
//! messages) and the main view with its header, tabs and periodic updates. The pages themselves are in `pages.rs`,
//! their texts and decisions in `text.rs`.
//!
//! The view is drawn from the shared-memory header each time: a row reads its field when it is rendered, so nothing
//! has to be kept in step by hand. A once-a-second check redraws the window only when something it shows changed.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::input::{InputEvent, InputState, NumberInput, NumberInputEvent, StepAction};
use gpui_kit::component::menu::{DropdownMenu as _, PopupMenuItem};
use gpui_kit::component::notification::Notification;
use gpui_kit::component::tab::{Tab as TabItem, TabBar};
use gpui_kit::component::tooltip::Tooltip;
use gpui_kit::component::{
    h_flex, v_flex, ActiveTheme as _, Disableable as _, Icon, IconName, Root, Sizable as _, Theme, ThemeMode, ThemeRegistry, TitleBar, WindowExt as _,
};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::Focusable as _;
use gpui_kit::{
    accesskit, div, px, size, Anchor, AnyElement, App, AppContext as _, AssetSource, Bounds, Context, Div, Element, ElementId, Entity, FocusHandle, FontWeight,
    GlobalElementId, InspectorElementId, InteractiveElement as _, IntoElement, LayoutId, MouseButton, ParentElement as _, Pixels, Position, Render, Role,
    SharedString, Stateful, StatefulInteractiveElement as _, Style, Styled as _, Subscription, Task, Text, Window, WindowBounds, WindowDecorations,
    WindowOptions,
};
use neural_forge_protocol::mapping::Mapping;
use neural_forge_protocol::ShmHeader;

use crate::instance::Activations;
use crate::text::{Liveness, ScopeEntry, TelemetrySample};

pub(crate) const WEBSITE: &str = "https://github.com/labj1987/neural-forge";
pub(crate) const ISSUES: &str = "https://github.com/labj1987/neural-forge/issues";

// ─────────────────────────────────────────────────────────────────────────────
//  Assets: the toolkit's component icons plus the one this app adds.
// ─────────────────────────────────────────────────────────────────────────────

gpui_kit::assets::icon_assets!(ExtraIcons, [Pencil]);

pub struct Assets;

impl AssetSource for Assets {
    fn load(&self, path: &str) -> gpui_kit::Result<Option<std::borrow::Cow<'static, [u8]>>> {
        if let Some(bytes) = ExtraIcons.load(path)? {
            return Ok(Some(bytes));
        }
        gpui_kit::assets::Assets.load(path)
    }

    fn list(&self, path: &str) -> gpui_kit::Result<Vec<SharedString>> {
        let mut paths = gpui_kit::assets::Assets.list(path)?;
        paths.extend(ExtraIcons.list(path)?);
        paths.sort();
        paths.dedup();
        Ok(paths)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Messages
// ─────────────────────────────────────────────────────────────────────────────

/// A message this long stays until it is closed; shorter ones go away on their own.
const LONG_TOAST: usize = 120;

/// What a message's button does.
pub(crate) type ToastAction = Rc<dyn Fn(&mut Window, &mut App)>;

/// Shows `text` as a message at the bottom of the window (plain text: a game, profile or file name is shown as it is).
pub(crate) fn toast(window: &mut Window, cx: &mut App, text: &str) {
    show_toast(window, cx, text, None, false);
}

/// A message with a button, which stays until it is closed.
pub(crate) fn toast_with_action(window: &mut Window, cx: &mut App, text: &str, button: &'static str, action: ToastAction) {
    show_toast(window, cx, text, Some((button, action)), true);
}

/// A message that stays until it is closed.
pub(crate) fn toast_kept(window: &mut Window, cx: &mut App, text: &str) {
    show_toast(window, cx, text, None, true);
}

/// False when the message could not be shown: a window has no place for messages until the toolkit's root view is in
/// it, which is after the main view is built (pushing one earlier panics in the toolkit). It goes to stderr then.
fn show_toast(window: &mut Window, cx: &mut App, text: &str, action: Option<(&'static str, ToastAction)>, keep: bool) -> bool {
    if window.root::<Root>().flatten().is_none() {
        eprintln!("neural-forge: {text}");
        return false;
    }
    let mut notification = Notification::new().message(text.to_string()).placement(Anchor::BottomCenter).autohide(!keep && text.chars().count() <= LONG_TOAST);
    if let Some((button, action)) = action {
        notification = notification.action(move |_, _, _| {
            let action = action.clone();
            Button::new("toast-action").small().label(button).on_click(move |_, window, cx| action(window, cx))
        });
    }
    window.push_notification(notification, cx);
    announce(text);
    window.refresh();
    true
}

// ─────────────────────────────────────────────────────────────────────────────
//  Screen readers. GPUI reports an element to assistive technology only when it has an id and a role, so the text,
//  dialog names and announcements below are spelled out; a bare string child is drawn but never read.
// ─────────────────────────────────────────────────────────────────────────────

/// Text a screen reader can read (a label node holding the text).
pub(crate) fn label(id: impl Into<ElementId>, text: impl Into<SharedString>) -> Text {
    Text::new(id.into(), text.into())
}

/// The toolkit gives a dialog no accessible name and has no way to set one, so everything in a dialog, its buttons
/// included, sits in a group named after the title: a screen reader says the name when the keyboard focus moves into
/// the group, which is when the dialog opens.
pub(crate) fn named_group(id: &'static str, name: impl Into<SharedString>) -> Stateful<Div> {
    v_flex().id(id).role(Role::Group).aria_label(name).gap_3()
}

// Every dialog is opened with `.overlay(false)`: the toolkit's dimming layer also covers the invisible margin a
// client-decorated window keeps around itself for its shadow, which shows as a dark frame around the window on KDE and
// GNOME. Without it a dialog is as modal as with it (it takes the input and the focus); gone are the dimming,
// click-outside-to-close and dragging the window by its title bar, for which the dialog's own title stands in
// (`dialog_title`).

/// Puts the keyboard focus on a dialog's main button when the dialog opens.
pub(crate) struct InitialFocus {
    pub(crate) handle: FocusHandle,
    pending: Cell<bool>,
}

impl InitialFocus {
    pub(crate) fn new(cx: &mut App) -> Rc<Self> {
        Rc::new(Self { handle: cx.focus_handle(), pending: Cell::new(true) })
    }

    /// Call from the dialog's builder, which runs on every frame: only the first call moves the focus.
    pub(crate) fn apply(&self, window: &mut Window, cx: &mut App) {
        if self.pending.replace(false) {
            self.handle.focus(window, cx);
        }
    }
}

/// The height a dialog's content may take before it scrolls: the toolkit cuts off a dialog that is taller than the
/// window instead of scrolling it, which would take the buttons out of reach.
pub(crate) fn dialog_room(window: &Window) -> Pixels {
    (window.viewport_size().height - px(DIALOG_CHROME)).max(px(60.))
}

/// What a dialog needs of the window's height besides its content.
const DIALOG_CHROME: f32 = 220.;

/// What Enter does in a dialog. The toolkit hands Enter to the dialog before the focused button sees it, so the
/// dialog looks at where the focus is and acts as that button would.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum EnterKey {
    Confirm,
    Cancel,
    Nothing,
}

pub(crate) fn enter_key(action_focused: bool, cancel_focused: bool) -> EnterKey {
    if action_focused {
        EnterKey::Confirm
    } else if cancel_focused {
        EnterKey::Cancel
    } else {
        EnterKey::Nothing
    }
}

/// A dialog's title, which is also the handle to move the window by while the dialog is open.
pub(crate) fn dialog_title(title: impl Into<SharedString>) -> impl IntoElement {
    div().w_full().on_mouse_down(MouseButton::Left, |_, window, _| window.start_window_move()).child(label("title", title))
}

/// The button row at the bottom of a dialog.
pub(crate) fn dialog_buttons() -> Div {
    h_flex().justify_end().gap_2().pt_1()
}

thread_local! {
    /// The newest message, for `Announcer`.
    static ANNOUNCEMENT: RefCell<SharedString> = RefCell::default();
}

/// Has the screen reader read `text` out (through the main window's `Announcer`).
fn announce(text: &str) {
    ANNOUNCEMENT.with(|current| {
        let mut current = current.borrow_mut();
        // The announcement is the change of the node's name, so the same message twice in a row needs a name that
        // differs: an invisible zero-width space is added or dropped.
        *current = if current.as_ref() == text { format!("{text}\u{200b}").into() } else { text.to_string().into() };
    });
}

/// A live region: an invisible node whose accessible name is the newest message, marked "polite" so a screen reader
/// reads each change out once it has finished speaking (the toolkit's own notifications are not announced).
struct Announcer(SharedString);

impl IntoElement for Announcer {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl Element for Announcer {
    type RequestLayoutState = ();
    type PrepaintState = ();

    fn id(&self) -> Option<ElementId> {
        Some("announcer".into())
    }

    fn source_location(&self) -> Option<&'static std::panic::Location<'static>> {
        None
    }

    fn a11y_role(&self) -> Option<Role> {
        Some(Role::Status)
    }

    fn write_a11y_info(&self, node: &mut accesskit::Node) {
        node.set_live(accesskit::Live::Polite);
        if !self.0.is_empty() {
            node.set_label(self.0.to_string());
        }
    }

    fn request_layout(&mut self, _: Option<&GlobalElementId>, _: Option<&InspectorElementId>, window: &mut Window, cx: &mut App) -> (LayoutId, ()) {
        // Out of the flow and with no size: it takes no room in the window.
        let style = Style { position: Position::Absolute, ..Style::default() };
        (window.request_layout(style, [], cx), ())
    }

    fn prepaint(&mut self, _: Option<&GlobalElementId>, _: Option<&InspectorElementId>, _: Bounds<Pixels>, _: &mut (), _: &mut Window, _: &mut App) {}

    fn paint(&mut self, _: Option<&GlobalElementId>, _: Option<&InspectorElementId>, _: Bounds<Pixels>, _: &mut (), _: &mut (), _: &mut Window, _: &mut App) {}
}

/// What runs after a dialog's choice, once the dialog is out of the way.
pub(crate) type Continuation = Box<dyn FnOnce(&mut Window, &mut App)>;

/// A question with Cancel and one action (and optionally a second one, left of it).
pub(crate) struct Question {
    pub id: &'static str,
    pub title: SharedString,
    pub body: SharedString,
    pub ok_text: &'static str,
    /// The action removes something: its button is the danger colour.
    pub danger: bool,
}

/// Asks `question`; the focus starts on Cancel. `on_ok` runs at most once, after the dialog has closed; cancelling
/// (the button, Escape, the close button) runs `on_cancel`, if any, instead.
pub(crate) fn ask(question: Question, on_ok: Continuation, on_cancel: Option<Continuation>, window: &mut Window, cx: &mut App) {
    let Question { id, title, body, ok_text, danger } = question;
    let ok = Rc::new(RefCell::new(Some(on_ok)));
    let cancelled = Rc::new(RefCell::new(on_cancel));
    let action_focus = cx.focus_handle();
    let cancel_focus = cx.focus_handle();
    let first = Cell::new(true);
    window.open_dialog(cx, move |dialog, window, cx| {
        if first.replace(false) {
            cancel_focus.focus(window, cx);
        }
        let confirm = {
            let (ok, cancelled) = (ok.clone(), cancelled.clone());
            move |window: &mut Window, cx: &mut App| {
                cancelled.borrow_mut().take();
                if let Some(callback) = ok.borrow_mut().take() {
                    window.defer(cx, move |window, cx| callback(window, cx));
                }
            }
        };
        // Any way out that is not the action: Cancel, Escape, the close button.
        let dismiss = {
            let (ok, cancelled) = (ok.clone(), cancelled.clone());
            move |window: &mut Window, cx: &mut App| {
                ok.borrow_mut().take();
                if let Some(callback) = cancelled.borrow_mut().take() {
                    window.defer(cx, move |window, cx| callback(window, cx));
                }
            }
        };
        let (enter, enter_cancel, on_close) = (confirm.clone(), dismiss.clone(), dismiss.clone());
        let (on_action, on_cancel) = (action_focus.clone(), cancel_focus.clone());
        let action = Button::new("ok").label(ok_text).track_focus(&action_focus).on_click(move |_, window, cx| {
            window.close_dialog(cx);
            confirm(window, cx);
        });
        dialog
            .title(dialog_title(title.clone()))
            .overlay(false)
            .w(px(480.))
            .on_ok(move |_, window, cx| {
                match enter_key(on_action.is_focused(window), on_cancel.is_focused(window)) {
                    EnterKey::Confirm => enter(window, cx),
                    EnterKey::Cancel => enter_cancel(window, cx),
                    EnterKey::Nothing => return false,
                }
                true
            })
            .on_close(move |_, window, cx| on_close(window, cx))
            .child(
                named_group(id, title.clone()).child(div().id("body").max_h(dialog_room(window)).overflow_y_scroll().child(label("text", body.clone()))).child(
                    dialog_buttons()
                        .child(Button::new("cancel").label("Cancel").track_focus(&cancel_focus).on_click(move |_, window, cx| {
                            window.close_dialog(cx);
                        }))
                        .child(if danger { action.danger() } else { action.primary() }),
                ),
            )
    });
}

/// Says something with one button; `on_close` runs once the dialog has closed, however it was closed.
pub(crate) fn inform(
    id: &'static str,
    title: SharedString,
    body: SharedString,
    button: &'static str,
    on_close: Option<Continuation>,
    window: &mut Window,
    cx: &mut App,
) {
    let closed = Rc::new(RefCell::new(on_close));
    let focus = InitialFocus::new(cx);
    window.open_dialog(cx, move |dialog, window, cx| {
        focus.apply(window, cx);
        let closed = closed.clone();
        dialog
            .title(dialog_title(title.clone()))
            .overlay(false)
            .w(px(520.))
            .on_ok(|_, _, _| true)
            .on_close(move |_, window, cx| {
                if let Some(callback) = closed.borrow_mut().take() {
                    window.defer(cx, move |window, cx| callback(window, cx));
                }
            })
            .child(
                named_group(id, title.clone()).child(div().id("body").max_h(dialog_room(window)).overflow_y_scroll().child(label("text", body.clone()))).child(
                    dialog_buttons().child(Button::new("close").label(button).track_focus(&focus.handle).on_click(|_, window, cx| window.close_dialog(cx))),
                ),
            )
    });
}

// ─────────────────────────────────────────────────────────────────────────────
//  Groups and rows
// ─────────────────────────────────────────────────────────────────────────────

/// A titled group of rows in one bordered box, as every page is made of.
pub(crate) fn group(
    id: &'static str,
    title: &str,
    description: Option<SharedString>,
    suffix: Option<AnyElement>,
    rows: Vec<AnyElement>,
    cx: &App,
) -> AnyElement {
    let (border, muted) = (cx.theme().border, cx.theme().muted_foreground);
    let mut head = v_flex().flex_1().min_w_0();
    if !title.is_empty() {
        head = head.child(div().font_weight(FontWeight::SEMIBOLD).child(label("title", title.to_string())));
    }
    if let Some(description) = description {
        head = head.child(div().text_sm().text_color(muted).child(label("description", description)));
    }
    let mut list = v_flex().w_full().border_1().border_color(border).rounded(cx.theme().radius).bg(cx.theme().secondary).overflow_hidden();
    for (i, row) in rows.into_iter().enumerate() {
        list = list.child(div().when(i > 0, |d| d.border_t_1().border_color(border)).child(row));
    }
    v_flex()
        .id(id)
        .role(Role::Group)
        .aria_label(title.to_string())
        .gap_2()
        .child(h_flex().gap_3().items_end().child(head).children(suffix))
        .child(list)
        .into_any_element()
}

/// One row: the title and what it means on the left, its controls on the right.
pub(crate) fn row(id: impl Into<ElementId>, title: impl Into<SharedString>, subtitle: impl Into<SharedString>, cx: &App) -> Stateful<Div> {
    let subtitle: SharedString = subtitle.into();
    let mut texts = v_flex().flex_1().min_w_0().child(label("title", title));
    if !subtitle.is_empty() {
        texts = texts.child(div().text_sm().text_color(cx.theme().muted_foreground).child(label("subtitle", subtitle)));
    }
    h_flex().id(id).w_full().min_h(px(50.)).px_3().py_2().gap_3().child(texts)
}

/// What picking a menu entry does, given its index.
pub(crate) type OnPick = Rc<dyn Fn(usize, &mut Window, &mut App)>;

/// A menu button showing the selected one of `options`; picking another calls `on_pick` with its index.
pub(crate) fn menu_button(id: impl Into<ElementId>, title: &str, options: Vec<SharedString>, selected: usize, enabled: bool, on_pick: OnPick) -> AnyElement {
    let shown = options.get(selected).cloned().unwrap_or_default();
    Button::new(id)
        .small()
        .label(shown.clone())
        .dropdown_caret(true)
        .disabled(!enabled)
        .accessibility_label(format!("{title}: {shown}"))
        .dropdown_menu(move |mut menu, _, _| {
            for (i, option) in options.iter().enumerate() {
                let on_pick = on_pick.clone();
                menu = menu.item(PopupMenuItem::new(option.clone()).checked(i == selected).on_click(move |_, window, cx| on_pick(i, window, cx)));
            }
            menu.max_h(px(420.)).scrollable(true)
        })
        .into_any_element()
}

/// An element with a tooltip.
pub(crate) fn with_tooltip(id: impl Into<ElementId>, tip: impl Into<SharedString>, child: impl IntoElement) -> AnyElement {
    let tip: SharedString = tip.into();
    div().id(id).child(child).tooltip(move |window, cx| Tooltip::new(tip.clone()).build(window, cx)).into_any_element()
}

thread_local! {
    /// Set while a spin field's text is being brought in line with its value, so the change is not taken for typing.
    static SYNCING: Cell<bool> = const { Cell::new(false) };
}

/// A number field with step buttons, showing one value: a header field (in the unit the row shows) or a dialog's own.
/// The value is read through `get` whenever the field is brought up to date and written through `set` on every edit.
#[derive(Clone)]
pub(crate) struct Spin {
    pub(crate) input: Entity<InputState>,
    get: Rc<dyn Fn() -> f64>,
    set: Rc<dyn Fn(f64)>,
    lower: f64,
    upper: f64,
    step: f64,
}

impl Spin {
    /// Makes the field in `cx`'s view; `after` runs there after each edit. Keep the subscriptions for as long as the
    /// field lives.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new<V: 'static>(
        window: &mut Window,
        cx: &mut Context<V>,
        get: Rc<dyn Fn() -> f64>,
        set: Rc<dyn Fn(f64)>,
        (lower, upper): (f64, f64),
        step: f64,
        after: fn(&mut V, &mut Window, &mut Context<V>),
    ) -> (Self, [Subscription; 2]) {
        let text = Self::text_for(step, get());
        let input = cx.new(|cx| InputState::new(window, cx).default_value(text));
        let spin = Self { input: input.clone(), get, set, lower, upper, step };
        let typed = {
            let spin = spin.clone();
            cx.subscribe_in(&input, window, move |view, _, event: &InputEvent, window, cx| {
                if SYNCING.with(Cell::get) {
                    return;
                }
                match event {
                    InputEvent::Change => spin.typed(cx),
                    // The text goes back to the value's own form (and to the value, if what was typed is no number).
                    InputEvent::Blur | InputEvent::PressEnter { .. } => spin.sync(true, window, cx),
                    InputEvent::Focus => return,
                }
                after(view, window, cx);
            })
        };
        let stepped = {
            let spin = spin.clone();
            cx.subscribe_in(&input, window, move |view, _, event: &NumberInputEvent, window, cx| {
                let NumberInputEvent::Step(action) = event;
                let by = match action {
                    StepAction::Increment => spin.step,
                    StepAction::Decrement => -spin.step,
                };
                (spin.set)((((spin.get)() + by).clamp(spin.lower, spin.upper) * 1e6).round() / 1e6);
                spin.sync(true, window, cx);
                after(view, window, cx);
            })
        };
        (spin, [typed, stepped])
    }

    /// The value as the field shows it: the decimals its step has, and more (up to 3) while the value has them.
    fn text_for(step: f64, value: f64) -> String {
        let digits = crate::text::spin_digits(step, value) as usize;
        format!("{value:.digits$}")
    }

    /// Text typed into the field: a number is written (within the bounds) unless it is what the field already shows
    /// for the current value, whose further decimals are kept that way.
    fn typed(&self, cx: &mut App) {
        let text = self.input.read(cx).value();
        let value = (self.get)();
        let digits = crate::text::spin_digits(self.step, value);
        if crate::text::kept_spin_value(&text, value, digits).is_some() {
            return;
        }
        if let Ok(typed) = text.trim().replace(',', ".").parse::<f64>() {
            if typed.is_finite() {
                (self.set)(typed.clamp(self.lower, self.upper));
            }
        }
    }

    /// Brings the text in line with the value: always with `force`, else only while the field is not being typed in.
    pub(crate) fn sync(&self, force: bool, window: &mut Window, cx: &mut App) {
        let value = (self.get)();
        if !value.is_finite() {
            return;
        }
        let text = Self::text_for(self.step, value);
        let state = self.input.read(cx);
        if state.value().as_ref() == text || (!force && state.focus_handle(cx).is_focused(window)) {
            return;
        }
        SYNCING.with(|s| s.set(true));
        self.input.update(cx, |state, cx| state.set_value(text, window, cx));
        SYNCING.with(|s| s.set(false));
    }

    /// The field, named `name` for screen readers.
    pub(crate) fn field(&self, name: impl Into<SharedString>) -> AnyElement {
        div().w(px(150.)).flex_shrink_0().child(NumberInput::new(&self.input).small().accessibility_label(name)).into_any_element()
    }
}

/// A header field: one of the `AtomicU32`s a setting is stored in.
pub(crate) type Get = fn(&ShmHeader) -> &AtomicU32;

// ─────────────────────────────────────────────────────────────────────────────
//  Start-up
// ─────────────────────────────────────────────────────────────────────────────

/// The families tried for the UI font, in the order GPUI itself falls back to them.
const UI_FONTS: [&str; 9] = ["Ubuntu", "Adwaita Sans", "Cantarell", "Noto Sans", "DejaVu Sans", "Liberation Sans", "Arial", "Helvetica", "FreeSans"];

/// The families tried for fixed-width text, the toolkit's own default first.
const MONO_FONTS: [&str; 10] = [
    "DejaVu Sans Mono",
    "Noto Sans Mono",
    "Liberation Mono",
    "Ubuntu Mono",
    "Ubuntu Sans Mono",
    "Adwaita Mono",
    "Source Code Pro",
    "Hack",
    "FreeMono",
    "Nimbus Mono PS",
];

/// The first of `families` that is installed.
fn first_installed(families: &[&'static str], installed: &[String]) -> Option<&'static str> {
    families.iter().copied().find(|family| installed.iter().any(|name| name == family))
}

/// Starts the toolkit, with installed families as its fonts. Left alone, the theme asks for GPUI's "system UI font",
/// a family no Linux desktop ships, and for a fixed-width family that many lack; GPUI and the toolkit then log
/// warnings and fall back to the first family they find. Naming those up front gives the same text and no warnings.
pub fn init(cx: &mut App) {
    let installed = cx.text_system().all_font_names();
    let ui_font = first_installed(&UI_FONTS, &installed);
    if let Some(ui_font) = ui_font {
        // The toolkit keeps a theme that is already there, and families set on it are used as they are.
        let mut theme = Theme::default();
        theme.font_family = ui_font.into();
        theme.mono_font_family = first_installed(&MONO_FONTS, &installed).unwrap_or(ui_font).into();
        cx.set_global(theme);
    }
    gpui_kit::init(cx);
    if ui_font.is_some() {
        // What the toolkit does for a theme it creates itself: its light and dark colours, then the mode.
        let registry = ThemeRegistry::global(cx);
        let light = registry.default_light_theme().clone();
        let dark = registry.default_dark_theme().clone();
        let theme = Theme::global_mut(cx);
        theme.light_theme = light;
        theme.dark_theme = dark;
        Theme::change(ThemeMode::Light, None, cx);
    }
}

/// The colours a developer forces with `NEURAL_FORGE_GUI_THEME=dark|light` (a headless session has no desktop
/// preference to follow, and the screenshots are the dark theme).
fn forced_theme() -> Option<ThemeMode> {
    match neural_forge_protocol::env::var("NEURAL_FORGE_GUI_THEME").as_deref() {
        Some("dark") => Some(ThemeMode::Dark),
        Some("light") => Some(ThemeMode::Light),
        _ => None,
    }
}

/// The window is drawn with the GPU, so with no usable Vulkan or OpenGL ES driver (not even Mesa's software ones)
/// there is nothing to draw an error in. Says so through whatever the desktop offers, then exits: a process left
/// behind with no window would also keep later launches from starting.
fn no_window(error: &str) -> ! {
    let has_display = ["WAYLAND_DISPLAY", "DISPLAY"].into_iter().any(|name| std::env::var_os(name).is_some_and(|value| !value.is_empty()));
    let message = crate::text::no_window_message(error, has_display);
    eprintln!("neural-forge: {message}");
    if has_display {
        for (program, args) in crate::text::ERROR_DIALOGS {
            let status = std::process::Command::new(program).args(*args).arg(&message).stdin(std::process::Stdio::null()).status();
            if status.is_ok_and(|status| crate::text::helper_showed(program, status.code())) {
                break;
            }
        }
    }
    std::process::exit(1);
}

/// Writes the settings changed a moment ago, which wait for their save (`shm::SAVE_DELAY`).
fn save_pending_settings() {
    if let Err(e) = crate::shm::flush_settings() {
        eprintln!("neural-forge: saving config.ini failed: {e}");
    }
}

pub fn build_ui(cx: &mut App, activations: Activations, install_error: Option<String>) {
    // With no display GPUI would run on with no window and no word about it.
    if ["WAYLAND_DISPLAY", "DISPLAY"].into_iter().all(|name| std::env::var_os(name).is_none_or(|value| value.is_empty())) {
        no_window("no display server to connect to");
    }
    let screen = cx.primary_display().map(|display| {
        let area = display.bounds().size;
        (f32::from(area.width) as i32, f32::from(area.height) as i32)
    });
    let opened_shm = crate::shm::Shm::open();
    // The error window is as small as its text.
    let (width, height) = match &opened_shm {
        Ok(_) => crate::text::default_window_size(screen),
        Err(_) => (560, crate::text::MIN_HEIGHT),
    };
    let options = WindowOptions {
        window_bounds: Some(WindowBounds::centered(size(px(width as f32), px(height as f32)), cx)),
        window_min_size: Some(size(px(crate::text::MIN_WIDTH as f32), px(crate::text::MIN_HEIGHT as f32))),
        app_id: Some(crate::APP_ID.to_string()),
        // The app draws its own title bar, so it looks and works the same on every desktop, including compositors
        // with no server-side decorations.
        window_decorations: Some(WindowDecorations::Client),
        ..TitleBar::window_options()
    };
    let prepare = |window: &mut Window, cx: &mut App| {
        window.set_window_title("Neural Forge");
        match forced_theme() {
            Some(mode) => Theme::change(mode, Some(window), cx),
            None => {
                // Follow the desktop's light/dark preference, now and when it changes.
                Theme::sync_system_appearance(Some(window), cx);
                window.observe_window_appearance(|window, cx| Theme::sync_system_appearance(Some(window), cx)).detach();
            }
        }
    };
    let window = match opened_shm {
        Ok(shm) => {
            let shm = shm.0;
            let opened = gpui_kit::open_window(options, cx, |window, cx| {
                prepare(window, cx);
                cx.new(|cx| MainView::new(shm, window, cx))
            });
            let (window, view) = match opened {
                Ok(opened) => opened,
                Err(e) => no_window(&format!("{e:#}")),
            };
            // What needs the window to be up: messages, and the dialogs a developer opens at start.
            let _ = window.update(cx, |_, window, cx| {
                view.update(cx, |view, cx| view.started(install_error, window, cx));
            });
            window
        }
        Err(error) => {
            let opened = gpui_kit::open_window(options, cx, |window, cx| {
                prepare(window, cx);
                cx.new(|cx| crate::pages::ErrorView::new(&error, cx))
            });
            match opened {
                Ok((window, _)) => window,
                Err(e) => no_window(&format!("{e:#}")),
            }
        }
    };

    cx.on_window_closed(|cx, _| {
        // However the window went (the title bar's close button removes it without asking `on_window_should_close`): a
        // change made in the last moment is still saved.
        save_pending_settings();
        if cx.windows().is_empty() {
            cx.quit();
        }
    })
    .detach();

    // A second launch asks this one to show its window, with the activation token the desktop gave that launch when
    // there is one: on Wayland only that raises the window (see activation.rs). Without a token, or off Wayland,
    // GPUI's own request is what there is; a Wayland compositor then marks the window as wanting attention instead of
    // raising it.
    cx.spawn(async move |cx| {
        while let Ok(token) = activations.recv().await {
            let _ = window.update(cx, |_, window, _| {
                let raised = token.as_deref().is_some_and(|token| crate::activation::activate(window, token));
                if !raised {
                    window.activate_window();
                }
            });
        }
    })
    .detach();
}

// ─────────────────────────────────────────────────────────────────────────────
//  The main view
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Tab {
    Model,
    Composition,
    Debug,
    Status,
    Setup,
}

impl Tab {
    pub(crate) const ALL: [Tab; 5] = [Tab::Model, Tab::Composition, Tab::Debug, Tab::Status, Tab::Setup];

    fn title(self) -> &'static str {
        match self {
            Tab::Model => "Model",
            Tab::Composition => "Composition",
            Tab::Debug => "Debug",
            Tab::Status => "Status",
            Tab::Setup => "Setup",
        }
    }

    fn icon(self) -> IconName {
        match self {
            Tab::Model => IconName::Cpu,
            Tab::Composition => IconName::GalleryVerticalEnd,
            Tab::Debug => IconName::Search,
            Tab::Status => IconName::Network,
            Tab::Setup => IconName::Settings,
        }
    }

    /// The tab `NEURAL_FORGE_GUI_OPEN` names.
    fn named(name: &str) -> Option<Tab> {
        Tab::ALL.into_iter().find(|tab| tab.title().eq_ignore_ascii_case(name))
    }
}

/// Everything the window shows that something else can change: read once a second, and the window is drawn again only
/// when it differs from what is shown.
#[derive(Clone, Default, PartialEq)]
pub(crate) struct Live {
    /// The persisted settings as the header holds them (every bound row's value).
    pub settings: Vec<(String, String)>,
    pub regions: Vec<neural_forge_protocol::HudRegion>,
    pub alive: bool,
    pub cache: String,
    pub cuts: String,
    /// The model runs after the upscaler in the running game: its two rows are shown.
    pub after_upscaler: bool,
    pub capture_in_game_frame: bool,
    pub layer: &'static str,
    pub placement: String,
    pub tested: (String, Option<String>),
    pub hdr: String,
    /// The "Settings for" menu: its entries, the one whose settings the header holds, and what the line below says.
    pub entries: Vec<ScopeEntry>,
    pub scope_at: usize,
    pub own: neural_forge_supervisor::game_settings::Section,
    pub summary: String,
}

/// The Status tab's telemetry: three lines, the two graph labels and the last seconds of samples.
#[derive(Default)]
pub(crate) struct Telemetry {
    pub model: &'static str,
    pub game: String,
    pub fps: String,
    pub frame_label: String,
    pub model_label: String,
    pub history: VecDeque<TelemetrySample>,
    last_frames: Option<(u64, Instant)>,
    live: Liveness,
}

pub struct MainView {
    pub(crate) shm: Arc<Mapping>,
    pub(crate) tab: Tab,
    focus_handle: FocusHandle,
    pub(crate) spins: HashMap<String, Spin>,
    spin_subs: HashMap<String, [Subscription; 2]>,
    pub(crate) live: Live,
    liveness: Liveness,
    /// The installed Steam games' app ids and names, for the "Settings for" menu.
    steam_names: Vec<(String, String)>,
    banner_closed: bool,
    pub(crate) telemetry: Telemetry,
    pub(crate) desktop_hdr: Option<String>,
    /// Each tab's scroll position.
    pub(crate) scroll: [gpui_kit::ScrollHandle; 5],
    /// How many more ticks apply `NEURAL_FORGE_GUI_SCROLL`.
    scroll_aid: u8,
    pub(crate) pages: crate::pages::PageState,
    _tasks: Vec<Task<()>>,
    _subs: Vec<Subscription>,
}

impl MainView {
    fn new(shm: Arc<Mapping>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let focus_handle = cx.focus_handle();
        let pages = crate::pages::PageState::new(window, cx);
        let mut view = Self {
            shm,
            tab: Tab::Model,
            focus_handle,
            spins: HashMap::new(),
            spin_subs: HashMap::new(),
            live: Live::default(),
            liveness: Liveness::default(),
            steam_names: Vec::new(),
            banner_closed: false,
            telemetry: Telemetry::default(),
            desktop_hdr: None,
            scroll: Default::default(),
            scroll_aid: 4,
            pages,
            _tasks: Vec::new(),
            _subs: Vec::new(),
        };
        view.add_setting_spins(window, cx);
        view.refresh_model();
        view.live = view.read_live();
        view.sync_hud_spins(window, cx);
        // First-run flow: no model means neural rendering can't work at all yet (the layer presents untouched frames,
        // which a first-time user would never notice): land on Setup, with a banner explaining why.
        if !view.pages.model_present {
            view.tab = Tab::Setup;
        }
        // Developer aid: `NEURAL_FORGE_GUI_OPEN=<tab>` selects that tab at start (`passes` and `uninstall` open their
        // dialogs, see `started`), so each can be screenshotted without synthetic input.
        match neural_forge_protocol::env::var("NEURAL_FORGE_GUI_OPEN").as_deref() {
            Some("uninstall") => view.tab = Tab::Setup,
            Some(name) => view.tab = Tab::named(name).unwrap_or(view.tab),
            None => {}
        }

        // Once a second: what the header and the settings files say now.
        view._tasks.push(cx.spawn_in(window, async move |this, cx| loop {
            cx.background_executor().timer(Duration::from_secs(1)).await;
            if this.update_in(cx, |this, window, cx| this.tick(window, cx)).is_err() {
                break;
            }
        }));
        // Five times a second: the telemetry samples (a redraw only while the Status tab shows them changing).
        view._tasks.push(cx.spawn_in(window, async move |this, cx| loop {
            cx.background_executor().timer(Duration::from_millis(u64::from(crate::text::TELEMETRY_INTERVAL_MS))).await;
            if this.update(cx, |this, cx| this.sample_telemetry(cx)).is_err() {
                break;
            }
        }));
        // Every 30 s, on a worker: the desktop's HDR state (a D-Bus call or the compositor's own tool) and the Proton
        // copies that are out of date.
        view._tasks.push(cx.spawn_in(window, async move |this, cx| loop {
            let read = cx
                .background_executor()
                .spawn(async {
                    let stale = neural_forge_supervisor::steam::roots()
                        .into_iter()
                        .find(|r| !r.flatpak)
                        .map(|r| neural_forge_supervisor::compat_tool::stale_copies(&r.path))
                        .unwrap_or_default();
                    (neural_forge_supervisor::desktop_hdr::read(), stale)
                })
                .await;
            let updated = this.update(cx, |this, cx| {
                let (hdr, stale) = read;
                let stale: Vec<(String, String)> = stale.iter().map(|copy| (copy.tool.name.clone(), copy.describe())).collect();
                if this.desktop_hdr.as_ref() != Some(&hdr) || this.pages.stale != stale {
                    this.desktop_hdr = Some(hdr);
                    this.pages.stale = stale;
                    cx.notify();
                }
            });
            if updated.is_err() {
                break;
            }
            cx.background_executor().timer(Duration::from_secs(30)).await;
        }));
        // The installed Steam games' names, read on a worker like the Setup tab's list.
        view._tasks.push(cx.spawn_in(window, async move |this, cx| {
            let found = cx
                .background_executor()
                .spawn(async {
                    use neural_forge_supervisor::steam;
                    steam::games(&steam::roots()).into_iter().map(|g| (g.appid, g.name)).collect::<Vec<_>>()
                })
                .await;
            let _ = this.update_in(cx, |this, window, cx| {
                this.steam_names = found;
                this.tick(window, cx);
            });
        }));
        view.scan_steam(window, cx);
        view.scan_protons(window, cx);

        // A change is saved a moment after the last one (`shm::SAVE_DELAY`), so a held step button is one write.
        let saver = cx.to_async();
        crate::shm::on_schedule(move |generation| {
            saver
                .spawn(async move |cx| {
                    cx.background_executor().timer(crate::shm::SAVE_DELAY).await;
                    crate::shm::save_due(generation);
                })
                .detach();
        });
        // A change made in the last moment before closing is still saved.
        window.on_window_should_close(cx, |_, _| {
            save_pending_settings();
            true
        });
        view.focus_handle.focus(window, cx);
        view
    }

    /// Once the window can show messages and dialogs.
    fn started(&mut self, install_error: Option<String>, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(message) = install_error {
            toast_kept(window, cx, &message);
        }
        {
            let handle = window.window_handle();
            let async_cx = cx.to_async();
            crate::shm::on_save_error(move |message| {
                let _ = handle.update(&mut async_cx.clone(), |_, window, cx| toast(window, cx, &message));
            });
        }
        match neural_forge_protocol::env::var("NEURAL_FORGE_GUI_OPEN").as_deref() {
            Some("passes") => crate::pages::open_pass_dialog(&self.shm, window, cx),
            Some("uninstall") => self.uninstall_clicked(window, cx),
            Some("about") => crate::pages::show_about_dialog(window, cx),
            _ => {}
        }
    }

    /// Adds a spin field under `key`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn add_spin(
        &mut self,
        key: String,
        get: Rc<dyn Fn() -> f64>,
        set: Rc<dyn Fn(f64)>,
        bounds: (f64, f64),
        step: f64,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let (spin, subs) = Spin::new(window, cx, get, set, bounds, step, |this: &mut Self, window, cx| this.changed(window, cx));
        self.spins.insert(key.clone(), spin);
        self.spin_subs.insert(key, subs);
    }

    pub(crate) fn remove_spin(&mut self, key: &str) {
        self.spins.remove(key);
        self.spin_subs.remove(key);
    }

    /// After something here changed the header: the rows are drawn from it again, and what else shows it follows.
    pub(crate) fn changed(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.live = self.read_live();
        self.sync_hud_spins(window, cx);
        for spin in self.spins.values() {
            spin.sync(false, window, cx);
        }
        cx.notify();
    }

    /// The once-a-second check: follows a change made with `shmctl`, a loaded profile, a game's session starting or
    /// another program, and the layer's status. Draws only when something shown changed.
    fn tick(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        // Developer aid, with `NEURAL_FORGE_GUI_OPEN`: `NEURAL_FORGE_GUI_SCROLL=<pixels>` shows the tab scrolled down that
        // far, for a screenshot of its lower part. Applied in the first seconds, while the page's lists still fill.
        if self.scroll_aid > 0 {
            self.scroll_aid -= 1;
            if let Some(down) = neural_forge_protocol::env::var("NEURAL_FORGE_GUI_SCROLL").and_then(|v| v.parse::<f32>().ok()) {
                let handle = &self.scroll[self.tab as usize];
                handle.set_offset(gpui_kit::point(px(0.), -px(down.max(0.)).min(handle.max_offset().y)));
                cx.notify();
            }
        }
        let live = self.read_live();
        if live != self.live {
            self.live = live;
            self.sync_hud_spins(window, cx);
            for spin in self.spins.values() {
                spin.sync(false, window, cx);
            }
            cx.notify();
        }
    }

    fn read_live(&mut self) -> Live {
        use neural_forge_supervisor::game_settings::{self, GameKey};
        let hdr = self.shm.header();
        let alive = self.liveness.alive(hdr);
        let refresh = hdr.cache_refresh_holds.load(Ordering::Relaxed);
        let cached = hdr.cache_cached_holds.load(Ordering::Relaxed);
        // A closed game's counters stay in the channel: only a live layer's are current.
        let cache = if !alive || (refresh == 0 && cached == 0) {
            "Not running".to_string()
        } else {
            format!(
                "{refresh} network frames, {cached} reused; {:.1}% rejected on the last reused frame; cache passes {:.3} ms",
                f64::from(hdr.cache_rejected.load(Ordering::Relaxed)) / 100.0,
                f32::from_bits(hdr.cache_gpu_ms_bits.load(Ordering::Relaxed))
            )
        };
        // Counted only while the model runs before the upscaler (the edit cache counts its frames then).
        let cuts = if !alive || refresh == 0 {
            "Not running".to_string()
        } else {
            match hdr.scene_cuts.load(Ordering::Relaxed) {
                1 => "1 frame where the picture changed completely: the model started afresh there".to_string(),
                n => format!("{n} frames where the picture changed completely: the model started afresh there"),
            }
        };
        let attached = hdr.layer_attached.load(Ordering::Relaxed) != 0;
        let layer = if alive {
            "active: processing the game's frames"
        } else if attached {
            "idle: a game attached, but no frames are being processed (loading screen, paused or closed)"
        } else {
            "no game attached yet"
        };
        let placement = crate::text::preupscale_label(
            alive,
            hdr.placement.load(Ordering::Relaxed),
            hdr.preupscale_state.load(Ordering::Relaxed),
            hdr.preupscale_width.load(Ordering::Relaxed),
            hdr.preupscale_height.load(Ordering::Relaxed),
            hdr.preupscale_misses.load(Ordering::Relaxed),
            &hdr.preupscale_reason(),
        );
        // The layer's own status line: the model running, or why it is not.
        let layer_reason = if alive { hdr.layer_reason() } else { String::new() };
        let placement = if layer_reason.is_empty() { placement } else { format!("{placement} -- {layer_reason}") };

        // The "Settings for" menu brought in line with the header: its entries (a game the header names that the menu
        // lacks is added), the entry whose settings it holds, and the summary.
        let sections = game_settings::load_all();
        let held = game_settings::held_game(hdr);
        // The game on the channel: by app id, or by the executable the layer names.
        let on_channel: Vec<GameKey> = GameKey::exe(&hdr.game_name()).into_iter().collect();
        let running = neural_forge_protocol::games::key_for_scope(hdr.running_game.load(Ordering::Relaxed), &sections, &on_channel);
        let extra: Vec<GameKey> = held.iter().chain(running.iter()).cloned().collect();
        let entries = crate::text::scope_entries(&self.steam_names, &game_settings::games_with_settings(&sections), &extra);
        let scope = hdr.settings_game();
        let scope_at = entries.iter().position(|e| e.game.as_ref().map_or(0, GameKey::scope) == scope).unwrap_or(0);
        let game = entries.get(scope_at).and_then(|e| e.game.clone());
        let own = game.as_ref().map(|g| game_settings::overrides(&sections, g)).unwrap_or_default();
        let shipped = game.as_ref().and_then(neural_forge_protocol::games::shipped).map_or(0, <[_]>::len);
        let running_now = alive && game.is_some() && game.as_ref().map_or(0, GameKey::scope) == hdr.running_game.load(Ordering::Relaxed);
        let summary = crate::text::scope_summary(game.as_ref(), own.len(), shipped, running_now);

        Live {
            settings: neural_forge_protocol::persist::snapshot(hdr).into_iter().collect(),
            regions: hdr.hud_settings().regions,
            alive,
            cache,
            cuts,
            after_upscaler: alive && hdr.native_running.load(Ordering::Relaxed) == 0,
            capture_in_game_frame: alive && crate::text::edits_in_game_frame(hdr.preupscale_state.load(Ordering::Relaxed)),
            layer,
            placement,
            tested: crate::text::tested_label(hdr.game_name().as_str(), neural_forge_supervisor::compat::for_header(hdr)),
            hdr: neural_forge_supervisor::shm_status::hdr_line(hdr, self.desktop_hdr.as_deref()),
            entries,
            scope_at,
            own,
            summary,
        }
    }

    /// The game whose settings are shown (`None`: all games').
    pub(crate) fn scope_game(&self) -> Option<neural_forge_supervisor::game_settings::GameKey> {
        self.live.entries.get(self.live.scope_at).and_then(|e| e.game.clone())
    }

    fn sample_telemetry(&mut self, cx: &mut Context<Self>) {
        let hdr = self.shm.header();
        let t = &mut self.telemetry;
        // The model on a second GPU (`NEURAL_FORGE_OFFLOAD_GPU`) runs on the finished frame, not before the upscaler:
        // its line is the layer's (`layer_reason`).
        let reason = hdr.layer_reason();
        let model = if hdr.native_running.load(Ordering::Relaxed) != 0 {
            "running before the upscaler"
        } else if reason.starts_with("second GPU: the model runs on") {
            "running on the second GPU, on the finished frame"
        } else {
            "not running before the upscaler"
        };
        // The name is only as current as the frames: a closed game leaves its name behind.
        let name = hdr.game_name();
        let active = t.live.alive(hdr);
        let game = match (name.is_empty(), active) {
            (_, false) => "none attached".to_string(),
            (true, true) => "attached".to_string(),
            (false, true) => name,
        };
        // The rate is over the time that really passed (a busy main loop runs the timer late).
        let frames = neural_forge_protocol::load64(&hdr.layer_frames_lo, &hdr.layer_frames_hi);
        let now = Instant::now();
        let rate = t.last_frames.and_then(|(prev, at)| {
            let secs = now.duration_since(at).as_secs_f64();
            (secs > 0.0).then(|| frames.saturating_sub(prev) as f64 / secs)
        });
        t.last_frames = Some((frames, now));
        let fps = match rate {
            Some(l) if active => format!("{l:.1} presents/s"),
            _ => "—".to_string(),
        };
        let model_ms = f32::from_bits(hdr.preupscale_network_gpu_ms_bits.load(Ordering::Relaxed));
        let sample = TelemetrySample {
            frame_ms: rate.filter(|&f| active && f > 0.5).map(|f| (1000.0 / f) as f32),
            model_ms: (active && hdr.native_running.load(Ordering::Relaxed) != 0 && model_ms.is_finite() && model_ms > 0.0).then_some(model_ms),
        };
        let ms = |v: Option<f32>| v.map_or_else(|| "—".to_string(), |v| format!("{v:.1} ms"));
        let frame_label = format!("Frame time {}", ms(sample.frame_ms));
        let model_label = format!("Model (GPU) {}", ms(sample.model_ms));
        // The graph moves only while it holds a sample: an idle window is not drawn again.
        let moving = sample.frame_ms.is_some() || sample.model_ms.is_some() || t.history.iter().any(|s| s.frame_ms.is_some() || s.model_ms.is_some());
        let texts = t.model != model || t.game != game || t.fps != fps || t.frame_label != frame_label || t.model_label != model_label;
        if t.history.len() == crate::text::TELEMETRY_SAMPLES {
            t.history.pop_front();
        }
        t.history.push_back(sample);
        (t.model, t.game, t.fps, t.frame_label, t.model_label) = (model, game, fps, frame_label, model_label);
        if self.tab == Tab::Status && (moving || texts) {
            cx.notify();
        }
    }

    /// Writes one setting's field and draws the rows again.
    pub(crate) fn set_bits(&mut self, name: &'static str, get: Get, bits: u32, window: &mut Window, cx: &mut Context<Self>) {
        crate::shm::write_bits(&self.shm, name, get(self.shm.header()), bits);
        self.changed(window, cx);
    }

    fn pick_tab(&mut self, tab: Tab, cx: &mut Context<Self>) {
        self.tab = tab;
        cx.notify();
    }

    fn tab_bar(&self, id: &'static str, cx: &mut Context<Self>) -> TabBar {
        let selected = Tab::ALL.iter().position(|&tab| tab == self.tab).unwrap_or(0);
        TabBar::new(id)
            .segmented()
            .selected_index(selected)
            .children(
                Tab::ALL
                    .into_iter()
                    .map(|tab| TabItem::new().aria_label(tab.title()).child(h_flex().gap_1p5().child(Icon::new(tab.icon()).small()).child(tab.title()))),
            )
            .on_click(cx.listener(|this, index: &usize, _, cx| this.pick_tab(Tab::ALL[(*index).min(Tab::ALL.len() - 1)], cx)))
    }
}

/// Below this width the tabs move from the header to a bar at the bottom of the window.
const HEADER_TABS_MIN_WIDTH: f32 = 760.;

impl Render for MainView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let wide = window.viewport_size().width >= px(HEADER_TABS_MIN_WIDTH);
        let mut title_bar = TitleBar::new().child(div().pl_2().text_sm().font_weight(FontWeight::MEDIUM).child("Neural Forge"));
        if wide {
            // Clicks on the tabs must not start a window drag.
            title_bar = title_bar.child(h_flex().on_mouse_down(MouseButton::Left, |_, _, cx| cx.stop_propagation()).child(self.tab_bar("tabs", cx)));
        }
        let title_bar = title_bar.child(
            h_flex().px_2().on_mouse_down(MouseButton::Left, |_, _, cx| cx.stop_propagation()).child(
                Button::new("about")
                    .ghost()
                    .small()
                    .icon(Icon::new(IconName::Info))
                    .tooltip("About")
                    .accessibility_label("About")
                    .on_click(|_, window, cx| crate::pages::show_about_dialog(window, cx)),
            ),
        );

        let mut root =
            v_flex().id("main").track_focus(&self.focus_handle).size_full().bg(cx.theme().background).text_color(cx.theme().foreground).child(title_bar);
        if !self.pages.model_present && !self.banner_closed {
            root = root.child(
                h_flex()
                    .id("banner")
                    .w_full()
                    .px_4()
                    .py_2()
                    .gap_3()
                    .justify_center()
                    .bg(cx.theme().warning.opacity(0.2))
                    .child(label("text", "The neural rendering model is missing -- extract it in Setup"))
                    .child(Button::new("open-setup").small().label("Open Setup").on_click(cx.listener(|this, _, _, cx| {
                        this.banner_closed = true;
                        this.pick_tab(Tab::Setup, cx);
                    }))),
            );
        }
        // The Setup tab edits no setting.
        if self.tab != Tab::Setup {
            root = root.child(self.scope_bar(cx));
        }
        root = root.child(self.page(cx));
        if !wide {
            root = root.child(h_flex().w_full().justify_center().py_1().border_t_1().border_color(cx.theme().border).child(self.tab_bar("tabs-bottom", cx)));
        }
        root.child(Announcer(ANNOUNCEMENT.with(|text| text.borrow().clone())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enter_acts_as_the_focused_button() {
        assert_eq!(enter_key(true, false), EnterKey::Confirm);
        assert_eq!(enter_key(false, true), EnterKey::Cancel);
        assert_eq!(enter_key(false, false), EnterKey::Nothing, "Enter on the close button or the text decides nothing");
    }

    #[test]
    fn fonts_are_the_first_installed_of_the_list() {
        let installed = vec!["DejaVu Sans".to_string(), "Noto Sans".to_string()];
        assert_eq!(first_installed(&UI_FONTS, &installed), Some("Noto Sans"));
        assert_eq!(first_installed(&UI_FONTS, &[]), None);
    }

    #[test]
    fn a_spin_field_shows_the_values_own_decimals() {
        assert_eq!(Spin::text_for(0.05, 0.5), "0.50");
        assert_eq!(Spin::text_for(1.0, 16.0), "16");
        assert_eq!(Spin::text_for(1.0, 16.5), "16.5");
        assert_eq!(Spin::text_for(0.01, 0.004), "0.004");
    }

    #[test]
    fn the_tabs_are_named_as_the_developer_aid_names_them() {
        assert_eq!(Tab::named("status"), Some(Tab::Status));
        assert_eq!(Tab::named("setup"), Some(Tab::Setup));
        assert_eq!(Tab::named("passes"), None);
    }
}
