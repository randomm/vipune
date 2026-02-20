//! Shared test utilities for config module tests.

use std::sync::Mutex;

/// Mutex to serialize environment variable tests and prevent race conditions.
pub static ENV_MUTEX: Mutex<()> = Mutex::new(());

/// Clean up environment variables used by vipune config.
pub fn cleanup_env_vars(vars: &[&str]) {
    for var in vars {
        #[allow(clippy::disallowed_methods)]
        std::env::remove_var(var);
    }
}
