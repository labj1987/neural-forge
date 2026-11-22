//! `cargo xtask test-apprun`: tests data/AppRun (the AppImage's entry point, a shell script on purpose: it runs before
//! anything else on a user's system) in a scratch AppDir, with stand-ins for ldd, ldconfig, the dynamic loader, the
//! dialog tools and the GUI:
//! - when the system has everything, the GUI runs with its arguments and exactly the environment AppRun was given;
//! - it does not run, and stderr says why (once per library, with the install command for the distribution), when a
//!   linked library is missing, glibc is too old, a symbol the GUI needs is undefined, or the libraries it loads at
//!   run time are missing (libwayland-client only on a Wayland session; Vulkan's or EGL's loader, either is enough);
//! - without ldd, or when ldd fails without saying why, the dynamic loader is asked;
//! - with a display, the first tool that can show the message does (zenity, kdialog, notify-send, the notification
//!   service over D-Bus, xmessage).
//!
//! It replaced scripts/test_apprun.sh, case for case; the stand-ins are still the `/bin/sh` texts they were.

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};

use regex::Regex;

use crate::sh::{self, Fail, Res};

const LINKED: &str = "\tlinux-vdso.so.1 (0x1)\n\tlibxcb.so.1 => /usr/lib/libxcb.so.1 (0x2)\n\tlibxkbcommon.so.0 => /usr/lib/libxkbcommon.so.0 (0x3)\n\tlibxkbcommon-x11.so.0 => /usr/lib/libxkbcommon-x11.so.0 (0x4)\n\tlibgcc_s.so.1 => /usr/lib/libgcc_s.so.1 (0x5)\n\tlibc.so.6 => /usr/lib/libc.so.6 (0x6)";

fn fail(message: String) -> Fail {
    Fail::new(1, format!("test_apprun: {message}"))
}

fn io(path: &Path, e: std::io::Error) -> Fail {
    fail(format!("{}: {e}", path.display()))
}

fn cache_line(lib: &str) -> String {
    format!("\t{lib} (libc6,x86-64) => /usr/lib/{lib}\n")
}

/// An ld.so cache listing: a first line, then these libraries.
fn cache(first: &str, libs: &[&str]) -> String {
    format!("{first}\n{}", libs.iter().map(|l| cache_line(l)).collect::<String>())
}

/// The scratch AppDir and its stand-ins.
struct Scratch {
    tmp: PathBuf,
    bash: PathBuf,
}

impl Scratch {
    fn path(&self, name: &str) -> PathBuf {
        self.tmp.join(name)
    }

    fn tmp(&self) -> String {
        self.tmp.to_string_lossy().into_owned()
    }

    fn write(&self, name: &str, text: &str) -> Res<()> {
        std::fs::write(self.path(name), text).map_err(|e| io(&self.path(name), e))
    }

    /// The lines, each ended by a newline (`printf '%s\n' ...`).
    fn lines(&self, name: &str, lines: &[&str]) -> Res<()> {
        self.write(name, &lines.iter().map(|l| format!("{l}\n")).collect::<String>())
    }

    fn read(&self, name: &str) -> String {
        String::from_utf8_lossy(&std::fs::read(self.path(name)).unwrap_or_default()).into_owned()
    }

    /// As `$(cat FILE)` gives it: without the trailing newlines.
    fn text(&self, name: &str) -> String {
        self.read(name).trim_end_matches('\n').to_string()
    }

    fn exists(&self, name: &str) -> bool {
        self.path(name).exists()
    }

    fn remove(&self, name: &str) {
        let _ = std::fs::remove_file(self.path(name));
    }

    fn executable(&self, name: &str, text: &str) -> Res<()> {
        self.write(name, text)?;
        std::fs::set_permissions(self.path(name), std::os::unix::fs::PermissionsExt::from_mode(0o755)).map_err(|e| io(&self.path(name), e))
    }

    /// A stand-in program on AppRun's PATH.
    fn stub(&self, name: &str, body: &str) -> Res<()> {
        self.executable(&format!("bin/{name}"), &format!("#!/bin/sh\n{body}\n"))
    }

    /// ldd -r prints $tmp/ldd with @BIN@ as the path it was given (the loader names the object that needs a symbol
    /// that way) and exits with $tmp/ldd-rc.
    fn stub_ldd(&self) -> Res<()> {
        let tmp = self.tmp();
        self.stub("ldd", &format!("[ \"$1\" = -r ] || exit 2; sed \"s#@BIN@#$2#g\" '{tmp}/ldd'; exit $(cat '{tmp}/ldd-rc')"))
    }

    fn reset(&self) -> Res<()> {
        self.lines("ldd", &[LINKED])?;
        self.write("ldd-rc", "0\n")?;
        self.write(
            "cache",
            &cache("9 libs found in cache `/etc/ld.so.cache'", &["libc.so.6", "libxcb.so.1", "libvulkan.so.1", "libEGL.so.1", "libwayland-client.so.0"]),
        )?;
        self.lines("loader", &[LINKED])?;
        self.remove("shown");
        self.remove("env");
        self.stub_ldd()
    }

    /// AppRun with only these variables (and PATH), the GUI given two arguments, one with a space. True when it exited
    /// with 0.
    fn run(&self, vars: &[(&str, &str)]) -> Res<bool> {
        for name in ["out", "err", "env"] {
            self.remove(name);
        }
        let create = |name: &str| std::fs::File::create(self.path(name)).map_err(|e| io(&self.path(name), e));
        let mut apprun = Command::new(&self.bash);
        apprun.env_clear().env("PATH", self.path("bin")).env("TMPDIR", self.path("tmpdir"));
        apprun.env("LIBCHECK_OS_RELEASE", self.path("ubuntu")).env("LIBCHECK_GLIBC_HAVE", "2.35").envs(vars.iter().copied());
        apprun.arg(self.path("app/AppRun")).args(["--flag", "a b"]).stdin(Stdio::inherit()).stdout(create("out")?).stderr(create("err")?);
        Ok(apprun.status().map_err(|e| sh::not_started(&apprun, e))?.success())
    }

    /// The run must have ended with 0.
    fn passes(&self, case: &str, vars: &[(&str, &str)]) -> Res<()> {
        if self.run(vars)? {
            Ok(())
        } else {
            Err(fail(format!("{case}: AppRun failed: {}", self.text("err"))))
        }
    }

    /// The run must not have ended with 0.
    fn fails(&self, case: &str, vars: &[(&str, &str)]) -> Res<()> {
        if self.run(vars)? {
            Err(fail(format!("{case}: AppRun exited 0")))
        } else {
            Ok(())
        }
    }

    fn ran(&self, case: &str) -> Res<()> {
        if self.text("out") != "ran:--flag\nran:a b" {
            return Err(fail(format!("{case}: the GUI did not run with its arguments: {}", (self.read("out") + &self.read("err")).trim_end_matches('\n'))));
        }
        if !self.read("err").is_empty() {
            return Err(fail(format!("{case}: AppRun wrote to stderr: {}", self.text("err"))));
        }
        Ok(())
    }

    fn refused(&self, case: &str) -> Res<()> {
        if !self.read("out").is_empty() {
            return Err(fail(format!("{case}: the GUI ran: {}", self.text("out"))));
        }
        if self.exists("env") {
            return Err(fail(format!("{case}: the GUI ran")));
        }
        let err = self.read("err");
        if !err.lines().any(|l| l.starts_with("Neural Forge: Neural Forge can't start")) {
            return Err(fail(format!("{case}: no message on stderr: {}", self.text("err"))));
        }
        if err.matches('\n').count() != 1 {
            return Err(fail(format!("{case}: more than one stderr line: {}", self.text("err"))));
        }
        Ok(())
    }

    fn says(&self, case: &str, what: &str) -> Res<()> {
        if self.read("err").contains(what) {
            Ok(())
        } else {
            Err(fail(format!("{case}: stderr lacks '{what}': {}", self.text("err"))))
        }
    }

    /// What the dialog stand-ins recorded: nothing may have been shown.
    fn nothing_shown(&self, case: &str) -> Res<()> {
        if self.exists("shown") {
            Err(fail(format!("{case}: {}", self.text("shown"))))
        } else {
            Ok(())
        }
    }

    /// A line of what was shown must match.
    fn shown(&self, case: &str, pattern: &str) -> Res<()> {
        let re = Regex::new(pattern).map_err(|e| fail(format!("{case}: {e}")))?;
        if self.read("shown").lines().any(|l| re.is_match(l)) {
            Ok(())
        } else {
            Err(fail(format!("{case}: got {}", self.text("shown"))))
        }
    }

    fn shown_once(&self, case: &str, what: &str) -> Res<()> {
        if self.read("shown").matches('\n').count() == 1 {
            Ok(())
        } else {
            Err(fail(format!("{case}: more than one {what}: {}", self.text("shown"))))
        }
    }
}

fn run() -> Res<ExitCode> {
    let root = crate::root();
    let dir = sh::mktemp()?;
    let bash = sh::which("bash").ok_or_else(|| fail("bash is not on PATH".into()))?;
    let t = Scratch { tmp: dir.path().to_path_buf(), bash };
    let tmp = t.tmp();
    for sub in ["app/usr/bin", "bin", "lib", "tmpdir"] {
        std::fs::create_dir_all(t.path(sub)).map_err(|e| io(&t.path(sub), e))?;
    }
    std::fs::copy(root.join("data/AppRun"), t.path("app/AppRun")).map_err(|e| io(&root.join("data/AppRun"), e))?;
    // The GUI stand-in: its arguments one per line, and its environment, sorted.
    t.executable("app/usr/bin/neural-forge", &format!("#!/bin/sh\nprintf \"ran:%s\\n\" \"$@\"\nenv | sort > \"{tmp}/env\"\n"))?;
    // Only the tools AppRun uses, so the machine's own ldd, zenity, kdialog or notify-send are never found.
    for tool in ["dirname", "readlink", "sort", "tail", "head", "sed", "cat", "env", "mktemp", "rm"] {
        let real = sh::which(tool).ok_or_else(|| fail(format!("{tool} is not on PATH")))?;
        std::os::unix::fs::symlink(&real, t.path(&format!("bin/{tool}"))).map_err(|e| io(&real, e))?;
    }
    // ldconfig -p prints $tmp/cache. The loader stand-in (LIBCHECK_LOADER) prints $tmp/loader.
    t.stub_ldd()?;
    t.stub("ldconfig", &format!("[ \"$1\" = -p ] || exit 2; cat '{tmp}/cache'"))?;
    t.executable("loader-stub", &format!("#!/bin/sh\nsed \"s#@BIN@#$1#g\" \"{tmp}/loader\"\n"))?;
    t.write("ubuntu", "ID=ubuntu\nID_LIKE=debian\n")?;
    t.write("fedora", "ID=fedora\n")?;
    t.write("gentoo", "ID=gentoo\n")?;
    let (fedora, gentoo, loader) = (format!("{tmp}/fedora"), format!("{tmp}/gentoo"), format!("{tmp}/loader-stub"));
    let loader = [("LIBCHECK_LOADER", loader.as_str())];

    // Everything is there: the GUI runs, and its environment is exactly the one AppRun got (the shells add only PWD,
    // SHLVL and _). No PATH change, no markers.
    t.reset()?;
    t.passes("all fine", &[("FOO", "x y"), ("XDG_RUNTIME_DIR", "/run/user/1"), ("WAYLAND_DISPLAY", "wayland-0"), ("LD_LIBRARY_PATH", "/opt/lib")])?;
    t.ran("all fine")?;
    let mut expected = vec![
        "FOO=x y".to_string(),
        "LD_LIBRARY_PATH=/opt/lib".to_string(),
        "LIBCHECK_GLIBC_HAVE=2.35".to_string(),
        format!("LIBCHECK_OS_RELEASE={tmp}/ubuntu"),
        format!("PATH={tmp}/bin"),
        format!("TMPDIR={tmp}/tmpdir"),
        "WAYLAND_DISPLAY=wayland-0".to_string(),
        "XDG_RUNTIME_DIR=/run/user/1".to_string(),
    ];
    expected.sort();
    let own = Regex::new(r"^(PWD|SHLVL|_|OLDPWD)=").unwrap();
    let env = t.read("env");
    let got: Vec<&str> = env.lines().filter(|l| !own.is_match(l)).collect();
    if got != expected {
        return Err(fail(format!("all fine: the environment changed: expected\n{}\ngot\n{}", expected.join("\n"), got.join("\n"))));
    }
    // A system library's own unresolved symbol, and one of another object, are not the GUI's business.
    t.lines("ldd", &[LINKED, "undefined symbol: xcb_private\t(/usr/lib/libxcb.so.1)", "undefined symbol: x\t(@BIN@-other)"])?;
    t.passes("foreign symbols", &[])?;
    t.ran("foreign symbols")?;

    // A linked library is missing (named twice by ldd): refused, named once, with the distribution's install command.
    t.reset()?;
    t.lines("ldd", &[LINKED, "\tlibxkbcommon-x11.so.0 => not found", "\tlibxkbcommon-x11.so.0 => not found"])?;
    t.write("ldd-rc", "1\n")?;
    t.fails("missing library", &[])?;
    t.refused("missing library")?;
    t.says("missing library", "these system libraries are missing: libxkbcommon-x11.so.0.")?;
    t.says("missing library", "Install them with: sudo apt install libxkbcommon-x11-0. Then start Neural Forge again.")?;
    t.fails("missing library (Fedora)", &[("LIBCHECK_OS_RELEASE", &fedora)])?;
    t.says("missing library (Fedora)", "Install them with: sudo dnf install 'libxkbcommon-x11.so.0()(64bit)'.")?;
    t.fails("missing library (Fedora Silverblue)", &[("LIBCHECK_OS_RELEASE", &fedora), ("LIBCHECK_OSTREE_BOOTED", &fedora)])?;
    t.says("missing library (Fedora Silverblue)", "Install them with: rpm-ostree install 'libxkbcommon-x11.so.0()(64bit)', and restart the computer.")?;
    t.fails("missing library (unknown system)", &[("LIBCHECK_OS_RELEASE", &gentoo)])?;
    t.says("missing library (unknown system)", "Install the packages that provide them with your package manager, then start Neural Forge again.")?;
    // A library the table does not know: the command that finds its package.
    t.lines("ldd", &[LINKED, "\tlibfoo.so.3 => not found", "\tlibxcb.so.1 => not found"])?;
    t.fails("unknown library", &[])?;
    t.says(
        "unknown library",
        "Install them with: sudo apt install libxcb1. Find the package that provides each of the others with: apt-file search libfoo.so.3.",
    )?;

    // glibc older than the build's: the newest version needed, and the system's.
    t.reset()?;
    t.lines(
        "ldd",
        &[
            LINKED,
            "@BIN@: /lib/libc.so.6: version `GLIBC_2.38' not found (required by @BIN@)",
            "@BIN@: /lib/libc.so.6: version `GLIBC_2.39' not found (required by @BIN@)",
        ],
    )?;
    t.write("ldd-rc", "1\n")?;
    t.fails("old glibc", &[])?;
    t.refused("old glibc")?;
    t.says("old glibc", "it needs glibc 2.39 or newer; this system has 2.35.")?;

    // Symbols the GUI needs that the system's libraries lack, in the loader's forms: refused, the first three named.
    t.reset()?;
    t.lines(
        "ldd",
        &[
            LINKED,
            "undefined symbol: xkb_a\t(@BIN@)",
            "undefined symbol: xkb_b, version V_1\t(@BIN@)",
            "symbol xkb_c version V_2 not defined in file libxkbcommon.so.0 with link time reference\t(@BIN@)",
            "undefined symbol: xkb_d\t(@BIN@)",
            "undefined symbol: xcb_private\t(/usr/lib/libxcb.so.1)",
        ],
    )?;
    t.fails("missing symbols", &[])?;
    t.refused("missing symbols")?;
    t.says("missing symbols", "the system's libraries are too old: they lack xkb_a, xkb_b, xkb_c and 1 more. Update the system")?;

    // The run-time libraries. No Vulkan and no EGL loader: refused, both named, either is enough.
    t.reset()?;
    t.write("cache", &cache("x libs", &["libc.so.6", "libwayland-client.so.0"]))?;
    t.fails("no graphics loader", &[])?;
    t.refused("no graphics loader")?;
    t.says("no graphics loader", "missing: libvulkan.so.1, libEGL.so.1. ")?;
    t.says("no graphics loader", "one of libvulkan.so.1 and libEGL.so.1 is enough")?;
    t.says("no graphics loader", "sudo apt install libvulkan1 libegl1")?;
    // Either one is enough, from the cache or from LD_LIBRARY_PATH.
    t.write("cache", &cache("x libs", &["libc.so.6", "libEGL.so.1"]))?;
    t.passes("EGL only", &[])?;
    t.ran("EGL only")?;
    t.write("cache", &cache("x libs", &["libc.so.6"]))?;
    t.write("lib/libvulkan.so.1", "")?;
    t.passes("Vulkan from LD_LIBRARY_PATH", &[("LD_LIBRARY_PATH", &format!("/nonexistent:{tmp}/lib"))])?;
    t.ran("Vulkan from LD_LIBRARY_PATH")?;
    // libwayland-client: needed on a Wayland session only.
    t.write("cache", &cache("x libs", &["libc.so.6", "libvulkan.so.1"]))?;
    t.passes("X11 without libwayland-client", &[("DISPLAY", ":0"), ("LIBCHECK_DIALOG", "none")])?;
    t.ran("X11 without libwayland-client")?;
    t.fails("Wayland without libwayland-client", &[("WAYLAND_DISPLAY", "wayland-0"), ("LIBCHECK_DIALOG", "none")])?;
    t.refused("Wayland without libwayland-client")?;
    t.says("Wayland without libwayland-client", "missing: libwayland-client.so.0.")?;
    t.says("Wayland without libwayland-client", "sudo apt install libwayland-client0")?;
    // A cache that does not list glibc (libraries kept elsewhere, as on NixOS): nothing is said about run-time libraries.
    t.write("cache", &cache("0 libs", &["libfoo.so.1"]))?;
    t.passes("no usable cache", &[("WAYLAND_DISPLAY", "wayland-0")])?;
    t.ran("no usable cache")?;

    // No ldd: the dynamic loader is asked (LIBCHECK_LOADER stands in for running the GUI under LD_TRACE_LOADED_OBJECTS).
    t.reset()?;
    std::fs::remove_file(t.path("bin/ldd")).map_err(|e| io(&t.path("bin/ldd"), e))?;
    t.passes("no ldd", &loader)?;
    t.ran("no ldd")?;
    t.lines("loader", &[LINKED, "\tlibxcb.so.1 => not found"])?;
    t.fails("no ldd, missing library", &loader)?;
    t.refused("no ldd, missing library")?;
    t.says("no ldd, missing library", "missing: libxcb.so.1.")?;
    // ldd fails without naming a problem: the loader is asked, and only what it names stops the GUI.
    t.reset()?;
    t.write("ldd", "ldd: exited with unknown exit code (139)\n")?;
    t.write("ldd-rc", "1\n")?;
    t.passes("failing ldd, clean loader", &loader)?;
    t.ran("failing ldd, clean loader")?;
    t.lines("loader", &[LINKED, "@BIN@: /lib/libc.so.6: version `GLIBC_2.40' not found (required by @BIN@)"])?;
    t.fails("failing ldd, loader names glibc", &loader)?;
    t.says("failing ldd, loader names glibc", "needs glibc 2.40 or newer")?;

    // The dialog: none without a display, or with LIBCHECK_DIALOG=none; else zenity; a zenity that cannot run (127)
    // falls through to kdialog; without kdialog, notify-send.
    t.reset()?;
    t.lines("ldd", &[LINKED, "\tlibxcb.so.1 => not found"])?;
    t.stub("zenity", &format!("echo zenity >> '{tmp}/shown'"))?;
    t.stub("kdialog", &format!("echo \"kdialog $2\" >> '{tmp}/shown'"))?;
    t.stub("notify-send", &format!("echo \"notify-send $*\" >> '{tmp}/shown'"))?;
    let (x11, wayland) = ([("DISPLAY", ":0")], [("WAYLAND_DISPLAY", "wayland-0")]);
    t.fails("no display", &[])?;
    t.nothing_shown("no display: a dialog was shown")?;
    t.fails("dialogs off", &[("WAYLAND_DISPLAY", "wayland-0"), ("LIBCHECK_DIALOG", "none")])?;
    t.nothing_shown("dialogs off: a dialog was shown")?;
    t.fails("zenity", &x11)?;
    if t.text("shown") != "zenity" {
        return Err(fail(format!("zenity: expected zenity alone, got {}", t.text("shown"))));
    }
    t.refused("zenity")?;
    // A zenity that cannot open the display ends with 1, as a closed dialog does, and says why on stderr: the next tool
    // shows the message. Its stderr is read once it has ended, not to the end of a pipe that something it left running
    // still holds; the temporary file it went to is removed.
    t.remove("shown");
    let sleep = sh::which("sleep").ok_or_else(|| fail("sleep is not on PATH".into()))?;
    t.stub("zenity", &format!("echo '(zenity:42): Gtk-WARNING **: cannot open display: :0' >&2; {} 5 & exit 1", sleep.display()))?;
    let started = std::time::Instant::now();
    t.fails("zenity without a display", &x11)?;
    if started.elapsed().as_secs() >= 4 {
        return Err(fail("zenity without a display: waited for what it left running".into()));
    }
    t.shown("zenity without a display", "^kdialog Neural Forge can.t start")?;
    let left: Vec<String> =
        std::fs::read_dir(t.path("tmpdir")).map(|d| d.flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect()).unwrap_or_default();
    if !left.is_empty() {
        return Err(fail(format!("zenity without a display: left {}", left.join(" "))));
    }
    t.remove("shown");
    t.stub("zenity", "exit 127")?;
    t.fails("kdialog", &x11)?;
    t.shown("kdialog", "^kdialog Neural Forge can.t start")?;
    t.shown_once("kdialog", "dialog")?;
    t.remove("shown");
    t.remove("bin/kdialog");
    t.fails("notify-send", &wayland)?;
    t.shown("notify-send (critical, no expiry)", "^notify-send -u critical -t 0 -a Neural Forge Neural Forge Neural Forge can.t start")?;
    t.shown_once("notify-send", "message")?;
    // A desktop with none of the three (a stock COSMIC): the notification service is asked over D-Bus, with the message
    // as the body; where that fails too (no service), xmessage on an X display.
    t.remove("shown");
    t.remove("bin/notify-send");
    t.stub("gdbus", &format!("echo \"gdbus $*\" >> '{tmp}/shown'"))?;
    t.stub("xmessage", &format!("echo xmessage >> '{tmp}/shown'"))?;
    t.fails("gdbus", &wayland)?;
    t.shown(
        "gdbus",
        r"^gdbus call --session --dest org.freedesktop.Notifications .*Notify -- 'Neural Forge' uint32 0 '' 'Neural Forge' 'Neural Forge can\\'t start.* @as \[\] \{'urgency': <byte 2>\} int32 0$",
    )?;
    t.shown_once("gdbus", "message")?;
    t.remove("shown");
    t.stub("gdbus", "exit 1")?;
    t.fails("xmessage", &x11)?;
    if t.text("shown") != "xmessage" {
        return Err(fail(format!("xmessage: got {}", t.text("shown"))));
    }
    t.remove("shown");
    t.fails("no tool", &wayland)?;
    t.nothing_shown("xmessage without an X display")?;
    println!("test_apprun: ok");
    Ok(ExitCode::SUCCESS)
}

pub fn main(_args: &[String]) -> ExitCode {
    sh::finish(run())
}
