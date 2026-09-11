//! Placeholder for `vipune doctor --fts` handler tests.
//!
//! The real handler test module (desync detection, --repair skip-vs-rebuild,
//! idempotency, -p scoping, locked-DB fast-fail, orphan detection) is
//! implemented in task-c of this issue. This stub keeps the
//! `mod doctor_fts_tests` declaration in `mod.rs` compiling so task-a's
//! clap-surface and parse tests can build independently in their own worktree.
//! Task-c replaces the entire file body.

#[cfg(test)]
mod placeholder {
    /// Marker test so the test module is never empty (clippy / dead-module
    /// hygiene). Task-c replaces this with the real handler test suite.
    #[test]
    fn test_placeholder_task_c_will_replace() {}
}
