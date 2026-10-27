//! `vipune project merge <from> <to>` handler.
//!
//! Moves all rows from one project_id to another within a single transaction.

use crate::commands::output::{MergeResponse, print_json};
use crate::errors::Error;
use std::path::Path;
use std::process::ExitCode;
use std::time::Duration;

/// Build the MCP restart notice text.
///
/// This is a pure function so tests can verify the message content without
/// capturing stderr.
pub(crate) fn build_mcp_restart_notice() -> String {
    "Note: If a vipune MCP server is running, it holds its project_id from startup. Restart the MCP server to see rows under the new project id.".to_string()
}

/// Run the project merge operation.
///
/// # Arguments
///
/// * `db_path` - Path to the SQLite database
/// * `from_project_id` - Source project id to move rows from
/// * `to_project_id` - Target project id to move rows to
/// * `json` - If true, output JSON; otherwise human-readable
///
/// # Errors
///
/// Returns error if the database cannot be opened or the merge fails.
pub fn handle_merge(
    db_path: &Path,
    from_project_id: &str,
    to_project_id: &str,
    json: bool,
) -> Result<ExitCode, Error> {
    // Open database
    let mut db = super::db_busy::open_db(db_path)?;

    // Set busy timeout to 0ms for fast-fail behavior on database locks
    super::db_busy::wrap_busy(db.set_busy_timeout(Duration::ZERO))?;

    let rows_moved =
        super::db_busy::wrap_busy(db.merge_project_ids(from_project_id, to_project_id))?;

    let response = MergeResponse {
        from: from_project_id.to_string(),
        to: to_project_id.to_string(),
        rows_moved,
    };

    if json {
        print_json(&response);
    } else {
        println!(
            "Merged {} row(s) from '{}' to '{}'",
            response.rows_moved, response.from, response.to
        );
    }

    // Warn that a running MCP server must be restarted to see the merged rows.
    // Only emitted when rows were actually moved — if nothing changed, no restart is needed.
    if response.rows_moved > 0 {
        eprintln!("\n{}", build_mcp_restart_notice());
    }

    Ok(ExitCode::SUCCESS)
}
