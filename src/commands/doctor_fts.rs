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

use crate::commands::output::{DoctorFtsProject, DoctorFtsResponse, print_json};
use crate::errors::{Error, LOCKED_DB_MESSAGE};
use crate::sqlite::Database;
use crate::sqlite::fts::FtsDesyncReport;
use rusqlite::{Connection, OpenFlags};
use std::collections::HashMap;
use std::path::Path;
use std::process::ExitCode;

/// A database-wide view of `memories` counts, keyed by project.
#[derive(Default)]
struct MemoryCounts {
    /// Total `memories` rows per project. A project with zero rows is absent
    /// (the contract `Database::row_counts_by_project` documents).
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
    // Open READ-ONLY — this diagnostic must not modify the DB. Any accidental
    // write fails with SQLITE_READONLY instead of silently corrupting data.
    //
    // A read-only open takes no write lock, so it does not contend with a
    // `BEGIN EXCLUSIVE` writer at all: this path fails fast on the first
    // statement. Only the read-write `Database::open` below (and
    // `migrate_model`'s) waits SQLite's default busy timeout, because it runs
    // `create_schema`/migrations before a busy timeout can be applied.
    let db = Database::from_conn(super::db_busy::wrap_busy(Connection::open_with_flags(
        db_path,
        OpenFlags::SQLITE_OPEN_READ_ONLY,
    ))?);

    let report: FtsDesyncReport = super::db_busy::wrap_busy(db.detect_fts_desync())?;
    // One aggregate pass per handler invocation: `build_response` only reads
    // the per-project counts, and nothing below re-writes `memories`.
    let counts = super::db_busy::wrap_busy(memory_counts(&db))?;

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
    let pre = Database::from_conn(super::db_busy::wrap_busy(Connection::open_with_flags(
        db_path,
        OpenFlags::SQLITE_OPEN_READ_ONLY,
    ))?);
    let pre_report: FtsDesyncReport = super::db_busy::wrap_busy(pre.detect_fts_desync())?;
    // ONE aggregate pass for the whole handler: `rebuild_fts` rewrites only
    // `memories_fts`, never `memories` (see its doc), so the pre-rebuild
    // per-project counts are still exact for the post-rebuild response and a
    // second (or third) `row_counts_by_project()` would rescan the table for
    // a value that cannot have changed.
    let counts = super::db_busy::wrap_busy(memory_counts(&pre))?;

    // No desync → skip the rebuild entirely, report zero actions, data_version untouched.
    if !pre_report.is_desynced() {
        let response = build_response(&pre_report, &counts, project_filter, true, 0);
        emit(&response, json);
        return Ok(ExitCode::SUCCESS);
    }

    // Desynced → open read-write with ZERO busy timeout (reindex fast-fail
    // pattern) so a locked DB errors immediately with the actionable message
    // instead of hanging.
    //
    // `Database::open` runs `create_schema`, so on a `BEGIN EXCLUSIVE` store
    // the lock is hit at the open itself — the point *before* any rebuild has
    // run — and that failure must keep the plain transient framing. Only a lock
    // hit by the rebuild statement itself means the index may be half-rebuilt,
    // which is why the recovery note is appended below and not here.
    let db = open_read_write(db_path)?;
    // Rebuild the FTS index in place. This is the FTS5 special command — it
    // re-reads every `memories` row into `memories_fts` and is the only rebuild the
    // repo uses. It does not touch `memories` data. Fast-fail on a lock: a
    // SQLITE_BUSY here aborts the statement, so the caller is told to re-run
    // the repair (the rebuild is idempotent) rather than left guessing.
    if let Err(e) = db.rebuild_fts().map_err(Error::from) {
        return Err(recovery_for(e));
    }

    // Post-rebuild: detect again on the same (still-open, read-write) connection.
    let post_report: FtsDesyncReport = super::db_busy::wrap_busy(db.detect_fts_desync())?;
    let actions = post_report.total_desynced() + pre_report.total_desynced();
    let response = build_response(&post_report, &counts, project_filter, true, actions);

    emit(&response, json);
    Ok(ExitCode::SUCCESS)
}

/// Open the database read-write (schema creation is a no-op on an existing DB),
/// then set a ZERO busy timeout so a contended *statement* fast-fails with the
/// actionable message.
///
/// Scope of "fast-fail" (issue #255 review): the ZERO timeout is applied *after*
/// `Database::open` returns, so the open's own statements — `create_schema` and
/// `run_migrations`, the latter taking a `BEGIN EXCLUSIVE` — run under SQLite's
/// default busy timeout and block ~5s before failing. Measured against an
/// external `BEGIN EXCLUSIVE` writer, `doctor --fts --repair` reports the
/// actionable locked message ~5.0s after the call; statements on the returned
/// handle (the rebuild among them) fail immediately. The same is true of every
/// other open in the CLI, read-only ones included: an open is not lock-free, and
/// neither is a read-only connection's *first* statement — a plain `SELECT` still
/// takes the shared read lock that a `BEGIN EXCLUSIVE` writer holds — so those
/// paths wait the same default 5s before the locked message appears. Only calls
/// on a handle whose timeout was already zeroed fast-fail.
fn open_read_write(db_path: &Path) -> Result<Database, Error> {
    let db = super::db_busy::open_db(db_path)?;
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

/// The locked-repair guidance: the standard busy message plus the re-run
/// command, since an interrupted rebuild leaves the index half-populated.
///
/// Composed from [`LOCKED_DB_MESSAGE`] rather than a literal so the locked
/// wording has one source (see `errors::LOCKED_DB_LEAD`).
pub(crate) fn recovery_note() -> String {
    format!(
        "{LOCKED_DB_MESSAGE} The FTS index may be half-rebuilt; re-run `vipune doctor --fts --repair` to complete it."
    )
}

/// The rebuild failure → what the operator is told. The single decision the
/// mid-rebuild arm makes, extracted so
/// `the_recovery_note_requires_the_locked_verdict` drives the same function
/// the handler calls rather than a copy of its `if` (issue #255 review).
///
/// Only the classified locked verdict earns the half-rebuilt note; any other
/// failure passes through untouched, so an unrelated error from this call
/// chain can never be told its index is half-rebuilt.
pub(crate) fn recovery_for(e: Error) -> Error {
    if e.is_locked() {
        Error::Config(recovery_note())
    } else {
        e
    }
}

/// Count `memories` rows per project on an already-open `Database` in a
/// single aggregate pass. Deliberately NOT a `COUNT(*)` per project id:
/// `Database::row_counts_by_project` is the aggregate sibling (the shape
/// `export_scan::scan_all_rows` documents for scans). The global total is
/// never reported by either renderer, so it is not computed.
fn memory_counts(db: &Database) -> Result<MemoryCounts, Error> {
    let mut counts = MemoryCounts::default();
    for (project_id, rows) in db.row_counts_by_project()? {
        *counts.by_project.entry(project_id).or_insert(0) += rows;
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
