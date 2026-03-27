//! Shared test utilities for config module tests.

use std::sync::Mutex;

/// Mutex to serialize environment variable tests and prevent race conditions.
pub static ENV_MUTEX: Mutex<()> = Mutex::new(());

/// Clean up environment variables used by vipune config.
pub fn cleanup_env_vars(vars: &[&str]) {
    for var in vars {
        unsafe {
            std::env::remove_var(var);
        }
    }
}

#[test]
fn test_cleanup_env_vars() {
    unsafe {
        std::env::set_var("VIPUNE_TEST_VAR", "test_value");
    }
    cleanup_env_vars(&["VIPUNE_TEST_VAR"]);
    assert!(std::env::var("VIPUNE_TEST_VAR").is_err());
}
