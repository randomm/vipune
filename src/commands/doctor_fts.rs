//! Placeholder for the `vipune doctor --fts` handler.
//!
//! The real handler (desync detection, --repair, and the DoctorFtsResponse
//! struct) is implemented in task-c of this issue. This stub keeps the
//! `mod doctor_fts` declaration in `mod.rs` compiling so task-a's clap-surface
//! and parse tests can build independently in their own worktree. Task-c
//! replaces the entire file body.

use crate::errors::Error;
use std::path::Path;
use std::process::ExitCode;

/// Entry point for `vipune doctor --fts` (task-c stub).
///
/// # Errors
///
/// Returns an `Error::Config` indicating the handler is not yet implemented
/// (placeholder stub — task-c fills this in).
pub fn handle_doctor_fts(
    _db_path: &Path,
    _project_filter: Option<&str>,
    _repair: bool,
    _json: bool,
) -> Result<ExitCode, Error> {
    Err(Error::Config(
        "doctor --fts handler is not implemented yet (placeholder stub for task-a clap surface)"
            .to_string(),
    ))
}
