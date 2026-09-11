//! Hook event handler adapter (issue #191, task-a).
//!
//! Thin command-surface module that reads the Claude Code JSON payload
//! from stdin, delegates to the `crate::hook` module (task-b), and
//! always exits 0 with empty stdout on any non-fatal failure.
//!
//! The hook path must never load the ONNX model.

use std::io::Read;
use std::process::ExitCode;

/// Read the full stdin payload and run the hook pipeline.
///
/// Always returns `ExitCode::SUCCESS` — the hook must never surface an
/// error to the agent mid-session (malformed stdin, DB lock, empty
/// extraction, unknown event all map to silent success).
///
/// # Errors
///
/// Returns `Err` only on catastrophic failure that the caller should
/// still map to exit 0.
#[allow(unused_variables)] // `config` reserved for task-b's DB path
pub fn handle_hook_event(
    config: &crate::config::Config,
    _json: bool,
) -> Result<ExitCode, crate::errors::Error> {
    let mut input = String::new();
    if std::io::stdin().read_to_string(&mut input).is_err() {
        return Ok(ExitCode::SUCCESS);
    }

    // Delegate to the hook module (task-b). The full pipeline:
    // 1. Parse payload (cwd, event-specific fields)
    // 2. detect_project_at(payload.cwd, None)
    // 3. Zero-LLM extractor (credential-blocked)
    // 4. Dedup check + insert_with_hash per candidate
    // 5. Any failure → exit 0, empty stdout
    //
    // task-b (src/hook/) provides the actual implementation.
    // This adapter is the command-surface entry point called by execute().
    let _ = input; // silence unused until task-b wires in

    Ok(ExitCode::SUCCESS)
}
