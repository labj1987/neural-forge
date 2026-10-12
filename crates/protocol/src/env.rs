//! Environment variables. Every variable this project reads is named `NEURAL_FORGE_*`.

/// `std::env::var(name)` with surrounding whitespace trimmed, as an `Option`: `None` when the
/// variable is unset, not UTF-8, or blank. A launch option written by hand often carries a stray
/// space (`NEURAL_FORGE_PREUPSCALE="off "`), which must not turn a valid value into an unknown one.
pub fn var(name: &str) -> Option<String> {
    let value = std::env::var(name).ok()?;
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

/// Whether `name` is set at all, to anything.
pub fn is_set(name: &str) -> bool {
    std::env::var_os(name).is_some()
}

/// True when the variable is exactly `1`.
pub fn flag(name: &str) -> bool {
    var(name).is_some_and(|v| v == "1")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_named_variable() {
        // Unique names: tests share the process environment.
        assert!(!is_set("NEURAL_FORGE_ENVTEST_B"));
        std::env::set_var("NEURAL_FORGE_ENVTEST_A", "new");
        assert_eq!(var("NEURAL_FORGE_ENVTEST_A").as_deref(), Some("new"));
        std::env::set_var("NEURAL_FORGE_ENVTEST_B", "1");
        assert!(is_set("NEURAL_FORGE_ENVTEST_B") && flag("NEURAL_FORGE_ENVTEST_B"));
        std::env::set_var("NEURAL_FORGE_ENVTEST_C", " off\t");
        assert_eq!(var("NEURAL_FORGE_ENVTEST_C").as_deref(), Some("off"), "values are trimmed");
        std::env::set_var("NEURAL_FORGE_ENVTEST_D", "  ");
        assert_eq!(var("NEURAL_FORGE_ENVTEST_D"), None, "blank is unset");
        assert!(is_set("NEURAL_FORGE_ENVTEST_D"));
        std::env::set_var("NEURAL_FORGE_ENVTEST_E", " 1 ");
        assert!(flag("NEURAL_FORGE_ENVTEST_E"));
    }
}
