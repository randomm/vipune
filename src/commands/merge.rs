//! `vipune project merge <from> <to>` handler.
//!
//! Moves all rows from one project_id to another within a single transaction.

use crate::errors::Error;
use crate::output::{MergeResponse, print_json};
use crate::sqlite::Database;
use std::path::Path;
use std::process::ExitCode;
use std::time::Duration;

/// Wrap a database error, converting SQLITE_BUSY into the actionable MCP-server message.
fn wrap_busy<T>(result: Result<T, Error>) -> Result<T, Error> {
    match result {
        Ok(v) => Ok(v),
        Err(Error::SqliteModule(msg)) if msg.contains("database is locked") => {
            Err(Error::Config(
                "Database is locked. Another process (likely the MCP server) is holding a lock. Stop the MCP server and retry.".to_string()
            ))
        }
        Err(e) => Err(e),
    }
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
    let mut db = Database::open(db_path).map_err(|e| {
        let err_msg = e.to_string();
        if err_msg.contains("database is locked") {
            return Error::Config(
                "Database is locked. Another process (likely the MCP server) is holding a lock. Stop the MCP server and retry.".to_string()
            );
        }
        Error::Config(err_msg)
    })?;

    // Set busy timeout to 0ms for fast-fail behavior on database locks
    db.set_busy_timeout(Duration::ZERO)?;

    let rows_moved = wrap_busy(
        db.merge_project_ids(from_project_id, to_project_id)
            .map_err(Error::from),
    )?;

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

    Ok(ExitCode::SUCCESS)
}
