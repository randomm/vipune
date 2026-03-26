//! Shared test utilities for config module tests.

use std::sync::Mutex;

/// Mutex to serialize environment variable tests and prevent race conditions.
pub static ENV_MUTEX: Mutex<()> = Mutex::new(());
