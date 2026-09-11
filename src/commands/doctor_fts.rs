//! `vipune doctor --fts` handler (Issue #193).
//!
//! Detects and surfaces silent FTS5 desync, where `memories` has rows but
//! `memories_fts` is empty or only partially populated, so the BM25 leg of hybrid
//! search cannot degrade to nothing without the user knowing.
//!
//! * **Detection** (`--fts`, no `--repair`): a READ-ONLY bidirectional rowid join
//!   (`Database::detect_fts_desync`). It never re-opens the DB read-write and never
//!   touches the hybrid-search hot path. A desync is reported as a `Warning:` line
//!   naming the affected project(s) and both row counts, and the command still exits
//!   0 like the other doctor modes — it is a warning, not a hard error.
//! * **Repair** (`--fts --repair`): runs the detection pre-check first. When the
//!   pre-check finds zero desync it SKIPS the rebuild entirely and reports zero
//!   actions, leaving `PRAGMA data_version` unchanged. When desync is found it opens
//!   the DB read-write with a ZERO busy timeout (reindex fast-fail pattern) and runs
//!   the FTS5 rebuild special command `INSERT INTO memories_fts(memories_fts)
//!   VALUES('rebuild')`. Repair is always GLOBAL and ignores `-p`.

use crate::errors::Error;
use crate::output::{DoctorFtsProject, DoctorFtsResponse, print_json};
use crate::sqlite::Database;
use crate::sqlite::fts::FtsDesyncReport;
use rusqlite::{Connection, OpenFlags};
use std::collections::HashMap;
use std::path::Path;
use std::process::ExitCode;

/// Wrap a `rusqlite::Error`, converting a locked-DB error into the actionable
/// MCP-server message. Duplicated from `doctor.rs` (its copy is not `pub`).
fn wrap_rusqlite_busy<T>(result: Result<T, rusqlite::Error>) -> Result<T, Error> {
    match result {
        Ok(v) => Ok(v),
        Err(e) if e.to_string().contains("database is locked") => Err(Error::Config(
            "Database is locked. Another process (likely the MCP server) is holding a lock. Stop the MCP server and retry.".to_string(),
        )),
        Err(e) => Err(Error::Config(e.to_string())),
    }
}

/// Wrap a `Result<T, crate::sqlite::Error>`, converting a locked-DB error into the
/// actionable `Error::Config` message.
fn wrap_sqlite_busy<T>(result: Result<T, crate::sqlite::Error>) -> Result<T, Error> {
    match result {
        Ok(v) => Ok(v),
        Err(e) if e.to_string().contains("database is locked") => Err(Error::Config(
            "Database is locked. Another process (likely the MCP server) is holding a lock. Stop the MCP server and retry.".to_string(),
        )),
        Err(e) => Err(e.into()),
    }
}

/// Wrap a `Result<T, Error>` (crate::errors::Error), converting a locked-DB
/// `SqliteModule` error into the actionable `Error::Config` message.
fn wrap_errors_busy<T>(result: Result<T, Error>) -> Result<T, Error> {
    match result {
        Ok(v) => Ok(v),
        Err(Error::SqliteModule(msg)) if msg.contains("database is locked") => Err(Error::Config(
            "Database is locked. Another process (likely the MCP server) is holding a lock. Stop the MCP server and retry.".to_string(),
        )),
        Err(e) => Err(e),
    }
}

/// A database-wide view of `memories` counts, keyed by project.
#[derive(Default)]
struct MemoryCounts {
    /// Total `memories` rows, globally.
    total: usize,
    /// Total `memories` rows per project.
    by_project: HashMap<String, usize>,
}

/// Run the `doctor --fts` handler.
///
/// # Arguments
///
/// * `db_path` - Path to the SQLite database
/// * `project_filter` - If `Some`, scope the under-population report to this project.
///   Orphan rows are always global and `--repair` always ignores this.
/// * `repair` - If true, run the pre-check then rebuild the FTS index on desync.
/// * `json` - If true, output JSON; otherwise human-readable.
///
/// # Errors
///
/// Returns an error if the database cannot be opened or queried. A locked database
/// yields the actionable MCP-server message.
#[allow(dead_code)] // called from task-a's dispatch arm in commands/mod.rs (separate workstream)
pub fn handle_doctor_fts(
    db_path: &Path,
    project_filter: Option<&str>,
    repair: bool,
    json: bool,
) -> Result<ExitCode, Error> {
    if repair {
        handle_doctor_fts_repair(db_path, project_filter, json)
    } else {
        handle_doctor_fts_detect(db_path, project_filter, json)
    }
}

/// Detection-only path: open READ-ONLY, run the rowid join, report.
fn handle_doctor_fts_detect(
    db_path: &Path,
    project_filter: Option<&str>,
    json: bool,
) -> Result<ExitCode, Error> {
    // Open READ-ONLY — this diagnostic must not modify the DB. Any accidental write
    // fails with SQLITE_READONLY instead of silently corrupting data.
    let db = Database::from_conn(wrap_rusqlite_busy(Connection::open_with_flags(
        db_path,
        OpenFlags::SQLITE_OPEN_READ_ONLY,
    ))?);

    let report: FtsDesyncReport = wrap_sqlite_busy(db.detect_fts_desync())?;
    let counts = wrap_errors_busy(memory_counts(&db))?;

    let response = build_response(&report, &counts, project_filter, false, 0);

    if json {
        print_json(&response);
    } else {
        print_human_detect(&response);
    }

    Ok(ExitCode::SUCCESS)
}

/// Repair path: open READ-ONLY for the pre-check; if desynced, re-open read-write
/// with a ZERO busy timeout and rebuild. `-p` is ignored (repair is global).
fn handle_doctor_fts_repair(
    db_path: &Path,
    project_filter: Option<&str>,
    json: bool,
) -> Result<ExitCode, Error> {
    // Pre-check: open READ-ONLY and run the rowid join.
    let pre = Database::from_conn(wrap_rusqlite_busy(Connection::open_with_flags(
        db_path,
        OpenFlags::SQLITE_OPEN_READ_ONLY,
    ))?);
    let pre_report: FtsDesyncReport = wrap_sqlite_busy(pre.detect_fts_desync())?;

    // No desync → skip the rebuild entirely, report zero actions, data_version untouched.
    if !pre_report.is_desynced() {
        let counts = wrap_errors_busy(memory_counts(&pre))?;
        let response = build_response(&pre_report, &counts, project_filter, true, 0);
        emit(&response, json);
        return Ok(ExitCode::SUCCESS);
    }

    // Desynced → open read-write with ZERO busy timeout (reindex fast-fail pattern)
    // so a locked DB errors immediately with the actionable message instead of hanging.
    let db = open_read_write(db_path)?;
    // Rebuild the FTS index in place. This is the FTS5 special command — it
    // re-reads every `memories` row into `memories_fts` and is the only rebuild the
    // repo uses. It does not touch `memories` data.
    db.rebuild_fts()?;

    // Post-rebuild: detect again on the same (still-open, read-write) connection.
    let post_report: FtsDesyncReport = wrap_sqlite_busy(db.detect_fts_desync())?;
    let counts = wrap_errors_busy(memory_counts(&db))?;
    let actions = post_report.total_desynced() + pre_report.total_desynced();
    let response = build_response(&post_report, &counts, project_filter, true, actions);

    emit(&response, json);
    Ok(ExitCode::SUCCESS)
}

/// Open the database read-write (schema creation is a no-op on an existing DB), then
/// set a ZERO busy timeout so a locked DB fast-fails with the actionable message.
fn open_read_write(db_path: &Path) -> Result<Database, Error> {
    let db = Database::open(db_path).map_err(|e| {
        let msg = e.to_string();
        if msg.contains("database is locked") {
            return Error::Config(
                "Database is locked. Another process (likely the MCP server) is holding a lock. Stop the MCP server and retry.".to_string(),
            );
        }
        Error::Config(msg)
    })?;
    db.set_busy_timeout(std::time::Duration::ZERO)?;
    Ok(db)
}

fn emit(response: &DoctorFtsResponse, json: bool) {
    if json {
        print_json(response);
    } else {
        print_human_detect(response);
    }
}

/// Count `memories` rows globally and per project on an already-open `Database`.
fn memory_counts(db: &Database) -> Result<MemoryCounts, Error> {
    let mut counts = MemoryCounts::default();
    for pid in db.list_all_project_ids()? {
        let c = db.count_rows_for_project(&pid)?;
        counts.total += c;
        *counts.by_project.entry(pid).or_insert(0) += c;
    }
    Ok(counts)
}

/// Assemble the `DoctorFtsResponse` from a detection report and memory counts.
///
/// `repaired`/`actions` are only meaningful on the `--repair` path; on the
/// detection-only path they are both `false`/`0`.
fn build_response(
    report: &FtsDesyncReport,
    counts: &MemoryCounts,
    project_filter: Option<&str>,
    repaired: bool,
    actions: usize,
) -> DoctorFtsResponse {
    let in_sync = !report.is_desynced();

    // Under-population report: when a project filter is set, scope to that project.
    // A project with zero desynced rows is reported in-sync (no error). Orphans are
    // always global regardless of the filter.
    let mut underpopulated: Vec<DoctorFtsProject> = Vec::new();
    let projects: Vec<String> = match project_filter {
        Some(filter) => {
            if report
                .underpopulated_by_project
                .get(filter)
                .copied()
                .unwrap_or(0)
                > 0
            {
                vec![filter.to_string()]
            } else {
                Vec::new()
            }
        }
        None => report.underpopulated_by_project.keys().cloned().collect(),
    };

    for pid in projects {
        let missing = report
            .underpopulated_by_project
            .get(&pid)
            .copied()
            .unwrap_or(0);
        let total = counts.by_project.get(&pid).copied().unwrap_or(0);
        underpopulated.push(DoctorFtsProject {
            project_id: pid.clone(),
            memory_rows: total,
            missing_from_fts: missing,
        });
    }
    underpopulated.sort_by(|a, b| a.project_id.cmp(&b.project_id));

    DoctorFtsResponse {
        in_sync,
        underpopulated_by_project: underpopulated,
        orphan_rows: report.orphans,
        total_desynced: report.total_desynced(),
        repaired,
        actions,
    }
}

/// Human-readable output. On desync, a line prefixed `Warning:` names each affected
/// project and both row counts, plus the global orphan count. On in-sync, a
/// reassuring single line.
fn print_human_detect(response: &DoctorFtsResponse) {
    if response.in_sync {
        if response.repaired {
            println!("FTS index is in sync. No rebuild needed (0 actions).");
        } else {
            println!("FTS index is in sync with the memories table.");
        }
        return;
    }

    for p in &response.underpopulated_by_project {
        println!(
            "Warning: project '{}' has {} memories row(s) missing from the FTS index ({} total).",
            p.project_id, p.missing_from_fts, p.memory_rows
        );
    }
    if response.orphan_rows > 0 {
        println!(
            "Warning: {} FTS row(s) have no matching memories row (orphan).",
            response.orphan_rows
        );
    }
    println!(
        "FTS index is desynced: {} total desynced row(s). Run 'vipune doctor --fts --repair' to rebuild.",
        response.total_desynced
    );
}
