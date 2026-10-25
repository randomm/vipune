//! `vipune backup` handler.
//!
//! Produces a consistent point-in-time copy of the memories database using
//! SQLite's Online Backup API (rusqlite `backup` feature). The copy is a
//! complete, queryable SQLite file — safe to store offsite or restore with
//! `vipune import`'s sibling commands regardless of journal state.
//!
//! Contract (issue #195):
//! - Operates on the resolved `config.database_path` (honours `--db-path`).
//! - Reports the actionable "Database is locked…" message when the source DB
//!   is held by another process. Statements after the open fail immediately
//!   (`busy_timeout = 0`, same shape as merge/reindex); the open itself waits
//!   SQLite's default busy timeout, as it runs schema creation and migrations
//!   before that timeout can be applied.
//! - Never re-embeds and never decodes embeddings — the copy is a page-level
//!   duplicate, so corrupt/NULL embedding blobs carry through byte-identically.

use crate::errors::Error;
use crate::output::{BackupResponse, print_json};
use crate::sqlite::Database;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;

/// Run the online backup.
///
/// Copies `source` to `destination` via `rusqlite::backup::Backup::run_to_completion`,
/// which drives `sqlite3_backup_step(-1)` until every page is copied. If the
/// destination file already exists it is replaced with a fresh, consistent copy.
///
/// Returns the destination file size in bytes.
///
/// # Errors
///
/// Returns an error if the source cannot be opened, the backup step fails, or
/// the produced copy fails the SQLite integrity check.
fn run_online_backup(source: &Path, destination: &Path) -> Result<u64, Error> {
    let source_db = Database::open(source).map_err(super::db_busy::open_db_error)?;

    // Fast-fail for every statement after this point (MCP server holding a
    // write lock, etc.). The Online Backup API returns raw rusqlite errors when
    // the source is in a hot-journal state (another connection holds
    // `BEGIN EXCLUSIVE` — the same shape as a running MCP server holding a
    // write lock), so both that path and the `crate::sqlite::Error` path are
    // classified by the shared `db_busy` helper. The `Database::open` above
    // precedes this timeout, so a contended open waits SQLite's default busy
    // timeout (~5s) before failing.
    super::db_busy::wrap_busy(source_db.set_busy_timeout(Duration::ZERO))?;

    // Fresh destination: the Online Backup API copies pages into the target
    // database; a pre-existing target would leave a partially-overwritten file
    // if the backup fails mid-way, which is not a valid SQLite file.
    if destination.exists() {
        std::fs::remove_file(destination)?;
    }

    let mut dest_conn = rusqlite::Connection::open(destination)?;
    // `Backup::new` calls `sqlite3_backup_init`, which fails with
    // `SQLITE_BUSY` when the source is in a hot-journal state (e.g. another
    // connection has `BEGIN EXCLUSIVE` — the same shape as a running MCP
    // server holding a write lock). The error surfaces as a raw `rusqlite::
    // Error`, so it goes through the shared `wrap_busy`, whose `From` bound
    // converts it at the storage boundary and renders the actionable
    // MCP-server message.
    let backup = super::db_busy::wrap_busy(
        rusqlite::backup::Backup::new(source_db.conn(), &mut dest_conn).map_err(Error::from),
    )?;
    // Drive `step(-1)` ("back up all remaining pages" per the SQLite docs) in
    // a manual loop until it returns `Done`. `run_to_completion` does not
    // accept a negative page count (it asserts `pages_per_step > 0`), so we
    // loop ourselves. `Busy`/`Locked` are transient states that the loop
    // re-tries; a 0 pause is fine for a file-local copy (no competing writer
    // is expected, and the source was opened with the same process and no
    // competing writer here).
    use rusqlite::backup::StepResult;
    loop {
        match backup.step(-1)? {
            StepResult::Done => break,
            StepResult::Busy | StepResult::Locked => std::thread::sleep(Duration::from_millis(1)),
            StepResult::More => {}
            // `StepResult` is marked `#[non_exhaustive]` by rusqlite so the
            // public API can add variants without breaking downstream code.
            // Treat any future variant as transient (retry) rather than fatal.
            _ => std::thread::sleep(Duration::from_millis(1)),
        }
    }
    drop(backup);
    drop(dest_conn);

    // Post-condition: the produced file must be a queryable, consistent copy.
    // `PRAGMA integrity_check` returns a single row "ok" for a healthy file and
    // one row per defect for a corrupted file. This confirms the "byte-complete
    // and queryable" acceptance criterion without needing to diff every row.
    let dest_check = rusqlite::Connection::open(destination)?;
    let mut defects = 0usize;
    {
        let mut stmt = dest_check.prepare("PRAGMA integrity_check")?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
        for row in rows {
            let msg = row?;
            if msg != "ok" {
                defects += 1;
            }
        }
    }
    drop(dest_check);
    if defects > 0 {
        return Err(Error::Config(format!(
            "Backup integrity check failed ({} defect(s)); destination: {}",
            defects,
            destination.display()
        )));
    }

    let bytes = std::fs::metadata(destination)?.len();
    Ok(bytes)
}

/// Run the backup command.
///
/// # Arguments
///
/// * `db_path` - Path to the source SQLite database (already resolved after
///   the `--db-path` CLI override is applied in `run()`)
/// * `output` - Optional explicit output path; defaults to `<source>-backup.<ext>`
///   alongside the source
/// * `json` - If true, output JSON; otherwise human-readable
///
/// # Errors
///
/// Returns error if the source cannot be opened, the backup fails, or the
/// produced file fails the integrity check.
pub fn handle_backup(db_path: &Path, output: Option<&Path>, json: bool) -> Result<ExitCode, Error> {
    let destination = resolve_destination(db_path, output);

    let bytes = run_online_backup(db_path, &destination)?;

    // Report row count from the produced backup (not the source) so the
    // number we report is exactly the number of rows in the file a user would
    // restore. Both are identical by construction (the backup is a page-level
    // copy), but reading from the destination also re-confirms the destination
    // is a queryable file at the moment we report success.
    let rows = {
        let dest = rusqlite::Connection::open(&destination)?;
        let count: i64 = dest.query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))?;
        count as usize
    };

    let response = BackupResponse {
        source: db_path.display().to_string(),
        destination: destination.display().to_string(),
        rows,
        bytes,
    };

    if json {
        print_json(&response);
    } else {
        println!(
            "Backed up {} rows ({} bytes) to {}",
            response.rows, response.bytes, response.destination
        );
    }

    Ok(ExitCode::SUCCESS)
}

/// Resolve the destination path for the backup file.
///
/// If the caller supplied an explicit path, use it; otherwise derive
/// `<source>-backup.<ext>` from the source path.
fn resolve_destination(source: &Path, output: Option<&Path>) -> PathBuf {
    output.map(Path::to_path_buf).unwrap_or_else(|| {
        let stem = source
            .file_stem()
            .map(|s| s.to_os_string())
            .unwrap_or_else(|| "memories".into());
        let ext = source
            .extension()
            .map(|e| e.to_os_string())
            .unwrap_or_else(|| "db".into());
        let parent = source.parent().map(Path::new).unwrap_or(Path::new("."));
        let mut buf = parent.to_path_buf();
        buf.push(format!(
            "{}-backup.{}",
            stem.to_string_lossy(),
            ext.to_string_lossy()
        ));
        buf
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_destination_default_ext() {
        let src = Path::new("/tmp/memories.db");
        let dest = resolve_destination(src, None);
        assert_eq!(dest, PathBuf::from("/tmp/memories-backup.db"));
    }

    #[test]
    fn test_resolve_destination_explicit_path() {
        let src = Path::new("/tmp/memories.db");
        let explicit = Path::new("/backups/memories.db");
        let dest = resolve_destination(src, Some(explicit));
        assert_eq!(dest, PathBuf::from("/backups/memories.db"));
    }

    #[test]
    fn test_resolve_destination_source_without_extension() {
        let src = Path::new("/tmp/memories");
        let dest = resolve_destination(src, None);
        assert_eq!(dest, PathBuf::from("/tmp/memories-backup.db"));
    }
}
