//! `vipune version` / `vipune --help` CLI contract (issue #330).
//!
//! `version` is a diagnostic: it reports the binary's name and version and
//! nothing else, so it must keep working when the database path, the config
//! file or the schema is unusable. Until #330 the binary failed here — `run()`
//! called `Config::load()`, `Config::ensure_directories()` and
//! `MemoryStore::new()` before dispatching, so an unusable path, a malformed
//! `VIPUNE_*` value or a future `PRAGMA user_version` turned `vipune version`
//! into exit 1 and *created* the database file it was only meant to report on.
//! The fix short-circuits `Commands::Version` above those three calls in
//! `src/main.rs`; these tests drive the compiled binary as a child process,
//! because every tier of the bug lives in the `run()` prologue and is invisible
//! to an in-process call of `handle_version`.
//!
//! Subprocess capture (rather than in-process fd swapping) is used for the same
//! reason as `reindex_cli_tests`: the test harness's own output reader conflicts
//! with in-process fd swaps. The child harness and the fixtures live in
//! [`child`]; the AC assertions live in [`contract`].
//!
//! The AC numbers live in the issue, so each `#[test]` names the behaviour it
//! pins and carries its AC in the doc comment. AC6's other half — hook events
//! and `mcp` are untouched by the reordering — is not re-pinned here:
//! `commands::hook_install_tests` and the `--db-path` wiring tests in
//! `commands::reindex_cli_tests` already drive those paths through the same
//! `run()` prologue, which the reordering leaves in place.

#![cfg(test)]

mod child;
mod contract;

use crate::commands::output::version_output;

/// The exact human-mode stdout `vipune version` must produce: the handler's own
/// name/version pair, rendered the way the handler renders it, with the
/// `println!` newline asserted rather than assumed.
pub(crate) fn expected_version_line() -> String {
    let response = version_output();
    format!("{} {}\n", response.name, response.version)
}
