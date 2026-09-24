//! Environment variables. Every variable this project reads is named `NEURAL_FORGE_*`.

/// `std::env::var(name)`, as an `Option`.
pub fn var(name: &str) -> Option<String> {
    std::env::var(name).ok()
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
    }
}
