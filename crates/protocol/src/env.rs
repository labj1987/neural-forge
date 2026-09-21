//! Environment variables, with the pre-0.1.77 spelling still honoured.
//!
//! Every variable this project reads is named `NEURAL_FORGE_*`. Releases up to 0.1.76 used
//! `NEURALFORGE_*` (no second underscore), and a game's Steam launch options, a user's
//! shell profile or an already-running helper may still carry those. Each lookup therefore
//! tries the new name first and falls back to the legacy one, so a mixed old/new set of
//! layer, helper and GUI keeps agreeing on the values.

const NEW_PREFIX: &str = "NEURAL_FORGE_";
const LEGACY_PREFIX: &str = "NEURALFORGE_";

/// The pre-0.1.77 spelling of `name` (`NEURAL_FORGE_SHM` -> `NEURALFORGE_SHM`).
pub fn legacy_name(name: &str) -> Option<String> {
    name.strip_prefix(NEW_PREFIX).map(|rest| format!("{LEGACY_PREFIX}{rest}"))
}

/// `std::env::var(name)`, falling back to the legacy spelling when `name` is unset.
pub fn var(name: &str) -> Option<String> {
    std::env::var(name).ok().or_else(|| legacy_name(name).and_then(|legacy| std::env::var(legacy).ok()))
}

/// Whether `name` (or its legacy spelling) is set at all, to anything.
pub fn is_set(name: &str) -> bool {
    std::env::var_os(name).is_some() || legacy_name(name).is_some_and(|legacy| std::env::var_os(legacy).is_some())
}

/// True when the variable is exactly `1`.
pub fn flag(name: &str) -> bool {
    var(name).is_some_and(|v| v == "1")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_name_wins_and_legacy_is_the_fallback() {
        // Unique names: tests share the process environment.
        std::env::set_var("NEURALFORGE_ENVTEST_A", "old");
        assert_eq!(var("NEURAL_FORGE_ENVTEST_A").as_deref(), Some("old"));
        std::env::set_var("NEURAL_FORGE_ENVTEST_A", "new");
        assert_eq!(var("NEURAL_FORGE_ENVTEST_A").as_deref(), Some("new"));
        assert!(is_set("NEURAL_FORGE_ENVTEST_B") == false);
        std::env::set_var("NEURALFORGE_ENVTEST_B", "1");
        assert!(is_set("NEURAL_FORGE_ENVTEST_B") && flag("NEURAL_FORGE_ENVTEST_B"));
        assert_eq!(legacy_name("HOME"), None);
    }
}
