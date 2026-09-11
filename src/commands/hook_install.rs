//! Hook installer/uninstaller command surface (issue #191, task-d).
//!
//! Stub for the command surface; task-d provides the full implementation
//! (settings.json merge, idempotency, uninstall via pre-mutation state).

use std::process::ExitCode;

/// Install vipune hook entries into `~/.claude/settings.json`.
///
/// Merges vipune's entries into the existing hooks map without clobbering
/// foreign entries. Idempotent on re-run. Records pre-mutation state.
///
/// Stub: full implementation in task-d.
#[allow(dead_code)] // wired via Commands::Hook dispatch; body in task-d
pub fn handle_hook_install(_json: bool) -> Result<ExitCode, crate::errors::Error> {
    Ok(ExitCode::SUCCESS)
}

/// Remove vipune hook entries from `~/.claude/settings.json`.
///
/// Uses the recorded pre-mutation state to remove only vipune's entries
/// and restore the prior file shape.
///
/// Stub: full implementation in task-d.
#[allow(dead_code)]
pub fn handle_hook_uninstall(_json: bool) -> Result<ExitCode, crate::errors::Error> {
    Ok(ExitCode::SUCCESS)
}
