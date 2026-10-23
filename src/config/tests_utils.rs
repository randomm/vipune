//! Shared test utilities for config module tests.
//!
//! Every in-crate test that writes a `VIPUNE_*` knob production code reads has
//! to serialise on [`ENV_MUTEX`] and clear the knobs it depends on both on
//! entry and on exit. The knob list, the clearing helper and the RAII guard
//! that does the exit-clearing all live here, next to the mutex they protect,
//! so there is exactly one copy of each (see `docs/testing.md`).

use std::collections::BTreeSet;
use std::sync::Mutex;

/// Mutex to serialize environment variable tests and prevent race conditions.
pub static ENV_MUTEX: Mutex<()> = Mutex::new(());

/// Every `VIPUNE_*` knob `Config::load()` parses — one per `apply_*_override`
/// in `config/env_parser.rs`.
///
/// This is the single authoritative list: `config::overrides` tests clear it,
/// `config::tests` clears it around `Config::load()`, and
/// `commands::reindex_cli_tests` removes it from spawned children. A test that
/// clears only part of it leaves a malformed value (`-1`, `abc`) in the process
/// environment for the next `Config::load()` to `unwrap()` on — which is how one
/// real assertion failure used to cascade into two or three unrelated panics.
/// `test_config_env_vars_match_env_parser` pins the list to the parser by name:
/// it scans `config/env_parser.rs` for every `VIPUNE_*` name it spells and
/// asserts the two sets are equal, plus a 1:1 count of `apply_*_override`.
pub const CONFIG_ENV_VARS: &[&str] = &[
    "VIPUNE_DATABASE_PATH",
    "VIPUNE_EMBEDDING_MODEL",
    "VIPUNE_SIMILARITY_THRESHOLD",
    "VIPUNE_RECENCY_WEIGHT",
    "VIPUNE_HYBRID",
    "VIPUNE_DECAY_REFRESH_DAYS",
    "VIPUNE_PROMOTION_THRESHOLD",
    "VIPUNE_PRUNE_RETRIEVAL_LIMIT",
    "VIPUNE_PRUNE_MIN_AGE_DAYS",
];

/// Clean up environment variables used by vipune config.
///
/// Only safe to call while holding [`ENV_MUTEX`] — `set_var`/`remove_var`
/// racing a concurrent `getenv` is the hazard, not the values.
pub fn cleanup_env_vars(vars: &[&str]) {
    for var in vars {
        unsafe {
            std::env::remove_var(var);
        }
    }
}

/// Clear every knob in [`CONFIG_ENV_VARS`].
pub fn cleanup_config_env_vars() {
    cleanup_env_vars(CONFIG_ENV_VARS);
}

/// RAII guard for tests that write config knobs read back by `Config::load()`
/// (directly or through a helper).
///
/// Takes [`ENV_MUTEX`] with poison recovery — a genuine assertion failure in one
/// test must not make every sibling panic with `PoisonError` on top of it — and
/// clears [`CONFIG_ENV_VARS`] on entry and again on `Drop`. The exit-clear is
/// what a trailing `cleanup_env_vars(...)` as the last statement of a test does
/// **not** give you: a panicking assertion skips it, and the malformed value
/// then leaks into whatever test the scheduler runs next.
///
/// The lock is held for the whole body; take it as the first statement.
#[must_use = "the guard is the lock: a discarded guard releases it and skips the exit-clear"]
pub struct ConfigEnvGuard {
    _lock: std::sync::MutexGuard<'static, ()>,
}

impl ConfigEnvGuard {
    #[must_use = "hold the guard for the whole test body; discarding it releases the lock immediately"]
    pub fn acquire() -> Self {
        let _lock = lock_env_mutex();
        cleanup_config_env_vars();
        Self { _lock }
    }
}

impl Drop for ConfigEnvGuard {
    fn drop(&mut self) {
        cleanup_config_env_vars();
    }
}

/// Take [`ENV_MUTEX`], recovering from poisoning.
///
/// A test that panics while holding the mutex poisons it; propagating that
/// turns one real failure into a target-wide one, so every acquisition recovers
/// rather than `unwrap()`ing.
#[must_use = "hold the guard while the environment is mutated; discarding it releases the lock immediately"]
pub fn lock_env_mutex() -> std::sync::MutexGuard<'static, ()> {
    ENV_MUTEX
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The authoritative knob list must stay in step with the parser it protects,
/// otherwise adding a 10th `apply_*_override` silently re-exposes every reader
/// above to an inherited malformed value.
#[test]
fn test_config_env_vars_match_env_parser() {
    const ENV_PARSER: &str = include_str!("env_parser.rs");

    // One `apply_*_override` per knob — the count catches two functions sharing
    // one knob name, which the name-set comparison below cannot see.
    let apply_fns = ENV_PARSER
        .lines()
        .filter(|line| {
            let trimmed = line.trim_start();
            trimmed.starts_with("pub fn apply_") && trimmed.contains("_override(")
        })
        .count();
    assert_eq!(
        CONFIG_ENV_VARS.len(),
        apply_fns,
        "`CONFIG_ENV_VARS` is out of step with the `apply_*_override` functions in \
         `config/env_parser.rs` — add (or remove) the matching knob"
    );

    // And every name matches by spelling, not just in number: a knob the parser
    // reads but the list omits is never cleared, and a knob the list clears but
    // the parser no longer reads is dead weight that hides the omission of a
    // real one.
    let parsed = vipune_knob_names(ENV_PARSER);
    let declared: BTreeSet<String> = CONFIG_ENV_VARS
        .iter()
        .map(|&name| name.to_string())
        .collect();
    assert_eq!(
        parsed, declared,
        "`CONFIG_ENV_VARS` and the `VIPUNE_*` names spelled in `config/env_parser.rs` \
         differ — left are the names the parser reads, right the names these helpers clear"
    );
}

/// Every `VIPUNE_*` name spelled out in `source`, found by plain byte scanning:
/// the `VIPUNE_` prefix followed by a run of `A-Z` / `_`. Doc comments and the
/// `VIPUNE_HYBRID` warning string repeat knob names, so the result is a set —
/// and a knob missing from the file is exactly the drift the test above wants.
fn vipune_knob_names(source: &str) -> BTreeSet<String> {
    const PREFIX: &[u8] = b"VIPUNE_";
    let bytes = source.as_bytes();
    let mut names = BTreeSet::new();
    for (at, _) in bytes
        .windows(PREFIX.len())
        .enumerate()
        .filter(|(_, window)| *window == PREFIX)
    {
        let mut end = at + PREFIX.len();
        while end < bytes.len() && (bytes[end].is_ascii_uppercase() || bytes[end] == b'_') {
            end += 1;
        }
        names.insert(String::from_utf8_lossy(&bytes[at..end]).into_owned());
    }
    names
}

#[test]
fn test_cleanup_env_vars() {
    // Mutates the process env, so it takes the same guard as every other
    // env-touching test in this target — `set_var` racing a sibling's `getenv`
    // is unsound regardless of the variable's name, and the sibling readers
    // here (`Config::default()` -> `dirs::home_dir()` -> `getenv("HOME")`,
    // `detect_project` -> `env::var("VIPUNE_PROJECT"/"VIPUNE_DEBUG")`) run on
    // other threads of this same process. `VIPUNE_TEST_VAR` is not a config
    // knob, so it is cleared explicitly rather than via the shared list.
    let _guard = lock_env_mutex();
    unsafe {
        std::env::set_var("VIPUNE_TEST_VAR", "test_value");
    }
    cleanup_env_vars(&["VIPUNE_TEST_VAR"]);
    assert!(std::env::var("VIPUNE_TEST_VAR").is_err());
}
