//! The Steam launch option that turns Neural Forge on for a game, and how it is folded into (and
//! taken back out of) whatever launch options the game already has.
//!
//! Neural Forge's part is environment variables in front of everything else: `NEURAL_FORGE_ENABLE=1`
//! (the layer manifest's `enable_environment`) and `NEURAL_FORGE_TARGET_EXE=<exe>[,<exe>...]`, the
//! game's own executables (`steam::target_for`), which the GUI and the CLI always write so that no
//! store launcher or helper process can take the effect. In front, not just before
//! `%command%`: a wrapper such as `gamemoderun %command%` takes `NAME=value` after it as a program
//! to run. [`launch_option`] (the string the Setup tab shows and copies) is [`merge`] applied to an
//! empty string, so the copied string and the written one cannot diverge.
//!
//! [`merge`] and [`strip`] are a pair: both are idempotent, both leave every token that is not
//! Neural Forge's where it is (other variables, wrappers, game arguments after `%command%`), and
//! `strip(merge(x))` is `x` for any `x` that already had `%command%` (spaces between tokens
//! normalised). A string without `%command%` is game arguments, which Steam appends to the
//! command; `merge` moves them behind an inserted `%command%`, where they stay after `strip`.
//! The merge/strip pairing follows DLSS5oneclick-forlinux's `src/platform/launch_options.rs`
//! (MIT; see ATTRIBUTION.md).

const COMMAND: &str = "%command%";
const ENABLE: &str = "NEURAL_FORGE_ENABLE=1";
const ENABLE_PREFIX: &str = "NEURAL_FORGE_ENABLE=";
const TARGET_PREFIX: &str = "NEURAL_FORGE_TARGET_EXE=";

/// The launch option for a game with no options of its own: `NEURAL_FORGE_ENABLE=1
/// [NEURAL_FORGE_TARGET_EXE=<exe>] %command%`. `target_exe` is trimmed; empty means none.
pub fn launch_option(target_exe: &str) -> String {
    merge("", target_exe)
}

/// Steam runs the launch option through a shell, so a value with a space or any other shell
/// character is single-quoted (a `'` inside becomes `'\''`). A comma (between target executables)
/// is not special to the shell.
pub fn shell_quote(value: &str) -> String {
    if !value.is_empty() && value.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | ',')) {
        value.to_string()
    } else {
        format!("'{}'", value.replace('\'', "'\\''"))
    }
}

/// Splits on whitespace the way the shell Steam hands the string to does, keeping each token's
/// text as written: a `'...'` or `"..."` stretch (and a backslash-escaped character) stays inside
/// its token.
pub fn split_tokens(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut started = false;
    let mut quote: Option<char> = None;
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        match (quote, c) {
            (Some(q), c) if c == q => {
                quote = None;
                cur.push(c);
            }
            (Some('"'), '\\') | (None, '\\') => {
                cur.push(c);
                if let Some(next) = chars.next() {
                    cur.push(next);
                }
                started = true;
            }
            (Some(_), c) => cur.push(c),
            (None, '\'' | '"') => {
                quote = Some(c);
                cur.push(c);
                started = true;
            }
            (None, c) if c.is_whitespace() => {
                if started {
                    out.push(std::mem::take(&mut cur));
                    started = false;
                }
            }
            (None, c) => {
                cur.push(c);
                started = true;
            }
        }
    }
    if started {
        out.push(cur);
    }
    out
}

fn ours(token: &str) -> bool {
    token.starts_with(ENABLE_PREFIX) || token.starts_with(TARGET_PREFIX)
}

/// Whether `token` is a shell assignment: `NAME=value`, NAME a variable name.
fn assignment(token: &str) -> bool {
    let Some((name, _)) = token.split_once('=') else { return false };
    let mut chars = name.chars();
    chars.next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_') && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// The assignments at the start of `before`, the only ones the shell sets for the command: after
/// the first other word (a wrapper such as `gamemoderun`), `NAME=value` is that word's argument.
fn leading(before: &[String]) -> &[String] {
    &before[..before.iter().take_while(|t| assignment(t)).count()]
}

/// The value the shell gives the variable `prefix` assigns: its last assignment in `lead`, outer
/// quotes taken off.
fn value_of<'t>(lead: &'t [String], prefix: &str) -> Option<&'t str> {
    let raw = lead.iter().rev().find_map(|t| t.strip_prefix(prefix))?;
    Some(['\'', '"'].iter().find_map(|q| raw.strip_prefix(*q).and_then(|v| v.strip_suffix(*q))).unwrap_or(raw))
}

/// The tokens before `%command%` and the ones after it; `None` without a `%command%`.
fn around_command(tokens: &[String]) -> Option<(&[String], &[String])> {
    let at = tokens.iter().position(|t| t == COMMAND)?;
    Some((&tokens[..at], &tokens[at + 1..]))
}

/// Whether `before` (the tokens before `%command%`) turns Neural Forge on: the leading assignments
/// set `NEURAL_FORGE_ENABLE` to 1, and none of Neural Forge's variables sits after a wrapper (where
/// the wrapper would run it as a program and the game would not start).
fn enables(before: &[String]) -> bool {
    let lead = leading(before);
    value_of(lead, ENABLE_PREFIX) == Some("1") && !before[lead.len()..].iter().any(|t| ours(t))
}

/// Whether `options` turns Neural Forge on: `NEURAL_FORGE_ENABLE=1` among the variables in front of
/// everything before `%command%` (the last of them when it is set twice), and none of Neural
/// Forge's variables after a wrapper.
pub fn is_enabled(options: &str) -> bool {
    let tokens = split_tokens(options);
    around_command(&tokens).is_some_and(|(before, _)| enables(before))
}

/// Whether `options` turns Neural Forge on ([`is_enabled`]) with a target executable
/// (`NEURAL_FORGE_TARGET_EXE=` with a value, among the same leading variables).
pub fn has_target(options: &str) -> bool {
    let tokens = split_tokens(options);
    around_command(&tokens).is_some_and(|(before, _)| enables(before) && value_of(leading(before), TARGET_PREFIX).is_some_and(|v| !v.trim().is_empty()))
}

/// `existing` with Neural Forge turned on. Returned unchanged (byte for byte) when it already is
/// ([`is_enabled`]), with the same target executable if one is given. Otherwise Neural Forge's
/// variables go first, replacing any of its own elsewhere before `%command%`; a target executable
/// already there is kept when `target_exe` is empty. A string without `%command%` is game arguments,
/// which go behind an inserted `%command%`, less any of Neural Forge's variables among them.
pub fn merge(existing: &str, target_exe: &str) -> String {
    let target_exe = target_exe.trim();
    let wanted_target = (!target_exe.is_empty()).then(|| format!("{TARGET_PREFIX}{}", shell_quote(target_exe)));
    let tokens = split_tokens(existing);
    let command = around_command(&tokens);
    let (before, after): (&[String], &[String]) = command.unwrap_or((&[], &tokens));
    let lead = leading(before);
    let current_target = lead.iter().rev().find(|t| t.starts_with(TARGET_PREFIX));
    if command.is_some() && enables(before) && wanted_target.as_ref().is_none_or(|w| current_target == Some(w)) {
        return existing.to_string();
    }
    let kept_target = current_target.or_else(|| before.iter().rev().find(|t| t.starts_with(TARGET_PREFIX))).cloned();
    let mut out = vec![ENABLE.to_string()];
    out.extend(wanted_target.or(kept_target));
    out.extend(before.iter().filter(|t| !ours(t)).cloned());
    out.push(COMMAND.to_string());
    out.extend(after.iter().filter(|t| command.is_some() || !ours(t)).cloned());
    out.join(" ")
}

/// `existing` with exactly what [`merge`] adds taken out (`NEURAL_FORGE_ENABLE=...` and
/// `NEURAL_FORGE_TARGET_EXE=...` before `%command%`). Returned unchanged when there is none of it;
/// a result of nothing but `%command%` becomes the empty string, Steam's "no launch options".
pub fn strip(existing: &str) -> String {
    let tokens = split_tokens(existing);
    let Some((before, after)) = around_command(&tokens) else {
        return existing.to_string();
    };
    if !before.iter().any(|t| ours(t)) {
        return existing.to_string();
    }
    let rest: Vec<String> = before.iter().filter(|t| !ours(t)).cloned().collect();
    if rest.is_empty() && after.is_empty() {
        return String::new();
    }
    let mut out = rest;
    out.push(COMMAND.to_string());
    out.extend(after.iter().cloned());
    out.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_setup_tab_string_is_unchanged() {
        assert_eq!(launch_option(""), "NEURAL_FORGE_ENABLE=1 %command%");
        assert_eq!(launch_option("   "), "NEURAL_FORGE_ENABLE=1 %command%");
        assert_eq!(launch_option("  GTA5_Enhanced.exe  "), "NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_TARGET_EXE=GTA5_Enhanced.exe %command%");
        assert_eq!(launch_option("My Game.exe"), "NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_TARGET_EXE='My Game.exe' %command%");
        assert_eq!(launch_option("it's.exe"), "NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_TARGET_EXE='it'\\''s.exe' %command%");
    }

    #[test]
    fn a_target_is_seen_only_with_the_switch_and_a_value() {
        assert!(has_target("NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_TARGET_EXE=a.exe,b.exe %command%"));
        assert!(has_target("NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_TARGET_EXE='My Game.exe' %command% -x"));
        assert!(!has_target("NEURAL_FORGE_ENABLE=1 %command%"));
        assert!(!has_target("NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_TARGET_EXE= %command%"));
        assert!(!has_target("NEURAL_FORGE_TARGET_EXE=a.exe %command%"), "not switched on");
        assert_eq!(launch_option("a.exe,b.exe"), "NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_TARGET_EXE=a.exe,b.exe %command%");
    }

    #[test]
    fn tokens_keep_quoted_stretches_together() {
        assert_eq!(split_tokens("A=\"x y\" B='p q' C=a\\ b  %command%  -x"), ["A=\"x y\"", "B='p q'", "C=a\\ b", "%command%", "-x"]);
        assert_eq!(split_tokens("E='it'\\''s' \"\""), ["E='it'\\''s'", "\"\""]);
        assert!(split_tokens(" \t ").is_empty());
    }

    #[test]
    fn merge_keeps_real_world_options_and_strip_restores_them() {
        for (existing, merged) in [
            ("MANGOHUD=1 PROTON_ENABLE_WAYLAND=1 %command% -dx12 -skipintro", "NEURAL_FORGE_ENABLE=1 MANGOHUD=1 PROTON_ENABLE_WAYLAND=1 %command% -dx12 -skipintro"),
            ("gamemoderun %command%", "NEURAL_FORGE_ENABLE=1 gamemoderun %command%"),
            ("WINEDLLOVERRIDES=\"dxgi=n,b;winmm=n,b\" DXVK_CONFIG_FILE='/home/a/My Games/dxvk.conf' %command% --launcher-skip", "NEURAL_FORGE_ENABLE=1 WINEDLLOVERRIDES=\"dxgi=n,b;winmm=n,b\" DXVK_CONFIG_FILE='/home/a/My Games/dxvk.conf' %command% --launcher-skip"),
            ("gamescope -W 2560 -H 1440 -- %command%", "NEURAL_FORGE_ENABLE=1 gamescope -W 2560 -H 1440 -- %command%"),
            ("%command% -DLSSFG", "NEURAL_FORGE_ENABLE=1 %command% -DLSSFG"),
            ("", "NEURAL_FORGE_ENABLE=1 %command%"),
        ] {
            assert_eq!(merge(existing, ""), merged, "{existing:?}");
            assert_eq!(strip(merged), existing, "{existing:?}");
            assert!(is_enabled(merged) && !is_enabled(existing));
        }
    }

    #[test]
    fn merge_and_strip_are_idempotent() {
        for existing in ["", "-dx12", "MANGOHUD=1 %command% -x", "NEURAL_FORGE_ENABLE=1 %command%", "a b %command%", "NEURAL_FORGE_ENABLE=0 %command%"] {
            for target in ["", "GTA5_Enhanced.exe", "My Game.exe"] {
                let once = merge(existing, target);
                assert_eq!(merge(&once, target), once, "{existing:?} {target:?}");
                let stripped = strip(&once);
                assert_eq!(strip(&stripped), stripped);
                assert!(!is_enabled(&stripped));
                assert_eq!(merge(&stripped, target), once, "{existing:?} {target:?}");
            }
        }
    }

    #[test]
    fn already_enabled_options_are_returned_byte_for_byte() {
        for s in ["MANGOHUD=1  NEURAL_FORGE_ENABLE=1   %command%   -x", "NEURAL_FORGE_ENABLE=1 %command% --launcher-skip"] {
            assert_eq!(merge(s, ""), s);
        }
        let with_target = "NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_TARGET_EXE=a.exe %command%";
        assert_eq!(merge(with_target, "a.exe"), with_target);
        assert_eq!(merge(with_target, ""), with_target, "an existing target is kept when none is given");
        assert_eq!(merge(with_target, "b.exe"), "NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_TARGET_EXE=b.exe %command%");
        for s in ["MANGOHUD=1 %command%", "-dx12", "", "%command%"] {
            assert_eq!(strip(s), s, "nothing of ours: unchanged");
        }
    }

    #[test]
    fn plain_game_arguments_move_behind_command() {
        assert_eq!(merge("-dx12 -skipintro", ""), "NEURAL_FORGE_ENABLE=1 %command% -dx12 -skipintro");
        assert_eq!(strip("NEURAL_FORGE_ENABLE=1 %command% -dx12 -skipintro"), "%command% -dx12 -skipintro");
        assert!(!is_enabled("NEURAL_FORGE_ENABLE=1"), "without %command% it is a game argument");
        assert!(!is_enabled("%command% NEURAL_FORGE_ENABLE=1"));
        assert!(!is_enabled("NEURAL_FORGE_ENABLE=0 %command%"));
    }

    #[test]
    fn a_misplaced_or_disabled_variable_of_ours_is_replaced() {
        assert_eq!(merge("gamemoderun NEURAL_FORGE_ENABLE=0 %command%", ""), "NEURAL_FORGE_ENABLE=1 gamemoderun %command%");
        assert_eq!(merge("gamemoderun NEURAL_FORGE_TARGET_EXE=x.exe %command%", ""), "NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_TARGET_EXE=x.exe gamemoderun %command%");
        assert_eq!(strip("NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_TARGET_EXE='My Game.exe' MANGOHUD=1 %command%"), "MANGOHUD=1 %command%");
    }

    #[test]
    fn a_variable_of_ours_after_a_wrapper_does_not_count_and_is_moved_in_front() {
        // gamemoderun would run `NEURAL_FORGE_ENABLE=1` as the program: the game does not start.
        let misplaced = "gamemoderun NEURAL_FORGE_ENABLE=1 %command%";
        assert!(!is_enabled(misplaced));
        assert_eq!(merge(misplaced, ""), "NEURAL_FORGE_ENABLE=1 gamemoderun %command%");
        let both = "gamemoderun NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_TARGET_EXE=x.exe %command%";
        assert!(!is_enabled(both) && !has_target(both));
        assert_eq!(merge(both, "x.exe"), "NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_TARGET_EXE=x.exe gamemoderun %command%");
        // In front of the wrapper as well as after it: still rebuilt, since the one after breaks the launch.
        assert_eq!(merge("NEURAL_FORGE_ENABLE=1 gamemoderun NEURAL_FORGE_ENABLE=1 %command%", ""), "NEURAL_FORGE_ENABLE=1 gamemoderun %command%");
        // Any other variable in front keeps ours in the leading run.
        assert!(is_enabled("MANGOHUD=1 NEURAL_FORGE_ENABLE=1 %command%"));
        assert!(has_target("MANGOHUD=1 NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_TARGET_EXE=a.exe %command%"));
        assert!(!has_target("NEURAL_FORGE_ENABLE=1 env NEURAL_FORGE_TARGET_EXE=a.exe %command%"));
    }

    #[test]
    fn the_last_assignment_is_the_value_and_ours_never_become_game_arguments() {
        // The shell keeps the last assignment: this one is off.
        let off = "NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_ENABLE=0 %command%";
        assert!(!is_enabled(off));
        assert_eq!(merge(off, ""), "NEURAL_FORGE_ENABLE=1 %command%");
        assert!(is_enabled("NEURAL_FORGE_ENABLE=0 NEURAL_FORGE_ENABLE=1 %command%"));
        assert!(is_enabled("NEURAL_FORGE_ENABLE='1' %command%"));
        // The last target is the one the layer sees.
        let two = "NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_TARGET_EXE=a.exe NEURAL_FORGE_TARGET_EXE=b.exe %command%";
        assert_eq!(merge(two, "b.exe"), two);
        assert_eq!(merge(two, "a.exe"), "NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_TARGET_EXE=a.exe %command%");
        // Ours typed without %command% are not passed to the game as arguments.
        assert_eq!(merge("NEURAL_FORGE_ENABLE=1", ""), "NEURAL_FORGE_ENABLE=1 %command%");
        assert_eq!(merge("-dx12 NEURAL_FORGE_TARGET_EXE=x.exe", ""), "NEURAL_FORGE_ENABLE=1 %command% -dx12");
        // After %command% they are the user's own game arguments, left where they are.
        assert_eq!(merge("%command% NEURAL_FORGE_ENABLE=1", ""), "NEURAL_FORGE_ENABLE=1 %command% NEURAL_FORGE_ENABLE=1");
    }
}
